import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from metrics import all_metrics

import tsgm

seed = 42

# set all seed fixed to ensure reproducibility
np.random.seed(seed)
tf.random.set_seed(seed)

# data and model settings
n_samples, seq_len, feat_dim, latent_dim, output_dim = 1024, 64, 32, 32, 2
max_value, const = 20, 10

# generate dataset
Xr, yr = tsgm.utils.gen_sine_vs_const_dataset(n_samples, seq_len, feat_dim, max_value, const)
scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))        
Xr = scaler.fit_transform(Xr).astype(np.float32) # scale data to [-1, 1]
yr = keras.utils.to_categorical(yr).astype(np.float32) # cast to one-hot encoding
ys = yr  # use real labels as synthetic labels

# real dataset
d_real = tsgm.dataset.Dataset(Xr, yr)

# training settings
epochs = 5
batch_size = 32
buffer_size = 128

dataset = tf.data.Dataset.from_tensor_slices((Xr, yr)) 
dataset = dataset.shuffle(buffer_size).batch(batch_size)


def get_cave(model_name):
    architecture = tsgm.models.zoo[model_name](seq_len, feat_dim, latent_dim, output_dim)
    encoder, decoder = architecture.encoder, architecture.decoder
    cvae = tsgm.models.cvae.cBetaVAE(encoder, decoder, latent_dim, temporal=False)
    cvae.compile(
        optimizer=keras.optimizers.Adam(),
    )
    cvae.fit(dataset, epochs=epochs)
    X_gen, y_gen = cvae.generate(ys)
    d_syn = tsgm.dataset.Dataset(X_gen, y_gen)
    return d_syn

def get_cgan(model_name):
   
    architecture = tsgm.models.zoo[model_name](seq_len, feat_dim, latent_dim, output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator
    cgan = tsgm.models.cgan.ConditionalGAN(discriminator, generator, latent_dim, temporal= False)

    cgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    cgan.fit(dataset, epochs=epochs)
    X_gen = cgan.generate(ys)
    y_gen = ys
    d_syn = tsgm.dataset.Dataset(X_gen, y_gen)
    return d_syn

def get_tcgan(model_name):
    # data
    X, y = tsgm.utils.gen_sine_const_switch_dataset(n_samples, seq_len, feat_dim, max_value, const)
    scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))
    X_train = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)
    tc_dataset = tf.data.Dataset.from_tensor_slices((X_train, y))
    tc_dataset = tc_dataset.shuffle(buffer_size).batch(batch_size)
    
    # model 
    architecture = tsgm.models.zoo[model_name](seq_len, feat_dim, latent_dim, output_dim=1) # for tcgan, output_dim shoule be 1
    discriminator, generator = architecture.discriminator, architecture.generator
    tcgan = tsgm.models.cgan.ConditionalGAN(discriminator, generator, latent_dim, temporal= True)
    
    tcgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    tcgan.fit(tc_dataset, epochs=epochs)
    
    tmp_latent = tf.random.normal(shape=(n_samples, seq_len, latent_dim))
    random_vector_labels = tf.concat(
        [tmp_latent, y[:n_samples, :, None]], axis=2
    )

    X_gen = tcgan.generator(random_vector_labels)
    y_gen = y
    
    d_syn_tc = tsgm.dataset.Dataset(X_gen, y_gen)
    d_real_tc = tsgm.dataset.Dataset(X_train, y)
    return d_real_tc, d_syn_tc, X_train, y
    


# five models are needed for the test
# "cvae_conv5": cVAE_CONV5Architecture
# "cgan_base_c4_l1": cGAN_Conv4Architecture
# "t-cgan_c4": tcGAN_Conv4Architecture
# "cgan_lstm_n": cGAN_LSTMnArchitecture
# "cgan_lstm_3": cGAN_LSTMConv3Architecture

def test_models():
    if keras.__version__.startswith("2."):
        print("Using keras 2, the metrics are as follows:")
    elif keras.__version__.startswith("3."):
        print("Using keras 3, the metrics are as follows:")

    cvae_dict = {    
        "cvae_conv5": "cVAE_CONV5Architecture",
    }
    for k, v in cvae_dict.items():
        d_syn = get_cave(k)
        all_metrics(d_real, d_syn, Xr, yr, v)
    
    cgan_dict = {
        "cgan_base_c4_l1": "cGAN_Conv4Architecture",
        "cgan_lstm_n": "cGAN_LSTMnArchitecture",
        "cgan_lstm_3": "cGAN_LSTMConv3Architecture"
    }
    for k, v in cgan_dict.items():
        d_syn = get_cgan(k)
        all_metrics(d_real, d_syn, Xr, yr, v)
        
    tcgan_dict = {
        "t-cgan_c4": "tcGAN_Conv4Architecture",
    }
    for k, v in tcgan_dict.items():
        d_real_tc, d_syn_tc, X_tc, y_tc = get_tcgan(k)
        all_metrics(d_real_tc, d_syn_tc, X_tc, y_tc,  v)
        
if __name__ == "__main__":
    test_models()