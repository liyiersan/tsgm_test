### Test codes of Conv1D for tsgm

This repo is the test code about applying layers.Conv1D in Keras3.0 for [tsgm](https://github.com/AlexanderVNikitin/tsgm).

The results can be seen in test_results.xlsx.

I found a significant performance difference between layers.Conv1D and LocallyConnected1D.

![image-20240509210204821](http://cdn.lisan.fun/img/image-20240509210204821.png)

According to the results, I guess that it may be inappropriate to directly replace LocallyConnected1D with layers.Conv1D.

#### How to prepare the environment

For tsgm with keras2.0:

```shell
pip install tsgm
```

or you can refer to https://github.com/AlexanderVNikitin/tsgm/issues/42

For tsgm with keras3.0:

```shell
pip install tensorflow
pip install tf_keras
cd tsgm_keras3_tf
python setup.py install
```

Note that the codes of `tsgm_keras3_tf/tsgm/models/architectures/locally_connected.py` is a little different from [keras-3-tf](https://github.com/AlexanderVNikitin/tsgm/blob/feature/keras-3-tf/tsgm/models/architectures/locally_connected.py) branch in line 15.

```python
# from keras import backend
from tensorflow.keras import backend
```

#### How to run

```shell
python test_tsgm.py
```

If you want to test keras3.0+LocallyConnected1D, you may need to modify the package source code in `zoo.py`.

Five models should be modified as follows:

```python
class cVAE_CONV5Architecture(BaseVAEArchitecture):
    
    def _build_decoder(self) -> keras.models.Model:
        # other codes
        d_output = LocallyConnected1D(self._feat_dim, 1, activation="sigmoid")(x)
        # d_output = layers.Conv1D(self._feat_dim, 1, activation="sigmoid")(x)
        decoder = keras.Model(inputs, d_output, name="decoder")
        return decoder
    
class cGAN_Conv4Architecture(BaseGANArchitecture):    
	def _build_generator(self) -> keras.models.Model:
        # other codes
        g_output = LocallyConnected1D(self._feat_dim, 1, activation="tanh")(x)
        # g_output = layers.Conv1D(self._feat_dim, 1, activation="tanh")(x)
        generator = keras.Model(g_input, g_output, name="generator")
        return generator
    
class tcGAN_Conv4Architecture(BaseGANArchitecture):
    def _build_generator(self) -> keras.models.Model:
        # other codes
        g_output = LocallyConnected1D(self._feat_dim, 1, activation="tanh")(x)
        # g_output = layers.Conv1D(self._feat_dim, 1, activation="tanh")(x)
        generator = keras.Model(g_input, g_output, name="generator")
        return generator
    
class cGAN_LSTMConv3Architecture(BaseGANArchitecture):
    def _build_generator(self) -> keras.models.Model:
        # other codes
        g_output = LocallyConnected1D(self._feat_dim, 1, activation="tanh")(x)
        # g_output = layers.Conv1D(self._feat_dim, 1, activation="tanh")(x)
        generator = keras.Model(g_input, g_output, name="generator")
        return generator

class cGAN_LSTMnArchitecture(BaseGANArchitecture):
    def _build_generator(self, output_activation: str) -> keras.Model:
        # other codes(x)
        g_output = LocallyConnected1D(self._feat_dim, 1, activation=output_activation)(x)
        # g_output = layers.Conv1D(self._feat_dim, 1, activation=output_activation)(x)
        generator = keras.Model(g_input, g_output, name="generator")
        return generator
```

#### Some confusion about `tsgm/tutorials/evaluation.ipynb`

###### 1. Use scaled data to build `d_real`

I find that the train_data for vae is scaled to [0, 1], which means that the generated data will be [0, 1] after training.

However, the `d_real` dataset takes unscaled `Xr` as the samples, which may lead to a high value when computing the distance. 

Therefore, I recommend to add a line after data normalization, just like this:

```python
# Using real data generate synthetic time series dataset
scaler = tsgm.utils.TSFeatureWiseScaler()        
scaled_data = scaler.fit_transform(Xr)

# update Xr to scaled_data
Xr = scaled_data
```

######  2. Update the codes of `DiscriminativeMetric`

In `evaluation.ipynb`, the **Discriminative metric** is computed as follows:

```python
# use LSTM classification model from TSGM zoo.
model = tsgm.models.zoo["clf_cl_n"](
    seq_len=Xr.shape[1], feat_dim=Xr.shape[2], output_dim=1).model
model.compile(
    tf.keras.optimizers.Adam(),
    tf.keras.losses.CategoricalCrossentropy(from_logits=False)
)

discr_metric = tsgm.metrics.DiscriminativeMetric()
print(
    discr_metric(
        d_hist=Xr, d_syn=Xs, model=model,
        test_size=0.2, random_seed=42, n_epochs=1
    )
)
```

It can be seen that the n_classes (*i.e.*, output_dim) of the classification model is set to 1.

I find that all classification models from TSGM zoo use softmax as the final activation. 

```python
m_output = layers.Dense(self._output_dim, activation="softmax")(x)
```

When n_classes is set to 1, the output would be all ones. This would result in the discriminative metric being very close to 0.5 since the classification model does not work.

But when I set n_classes to 2, some errors occur. I tried to add a breakpoint and debug, but I failed. I guess the reasons the error occurs may be follows:

```python
 	y_pred = (model.predict(X_test) > 0.5).astype(int)
    if metric is None:
        return sklearn.metrics.accuracy_score(y_test, y_pred)
    else:
        return metric(y_test, y_pred)
```

If n_classes is 2, y_pred will be a 2D array, which will cause the shape mismatch error in accuracy_score.









