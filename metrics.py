import numpy as np
import functools
import sklearn
import tensorflow as tf
import keras
import tsgm

def distance_metric(d_real, d_syn):
    statistics = [
        functools.partial(tsgm.metrics.statistics.axis_max_s, axis=None),
        functools.partial(tsgm.metrics.statistics.axis_min_s, axis=None),
        functools.partial(tsgm.metrics.statistics.axis_max_s, axis=1),
        functools.partial(tsgm.metrics.statistics.axis_min_s, axis=1)]
    discrepancy_func = lambda x, y: np.linalg.norm(x - y)
    dist_metric = tsgm.metrics.DistanceMetric(
        statistics=statistics, discrepancy=discrepancy_func
    )
    return dist_metric(d_real, d_syn)

def MMD_metric(Xr, Xs):
    mmd_metric = tsgm.metrics.MMDMetric()
    return mmd_metric(Xr, Xs)

def distance_metric(d_real, d_syn):
    statistics = [
        functools.partial(tsgm.metrics.statistics.axis_max_s, axis=None),
        functools.partial(tsgm.metrics.statistics.axis_min_s, axis=None),
        functools.partial(tsgm.metrics.statistics.axis_max_s, axis=1),
        functools.partial(tsgm.metrics.statistics.axis_min_s, axis=1)]
    discrepancy_func = lambda x, y: np.linalg.norm(x - y)
    dist_metric = tsgm.metrics.DistanceMetric(
        statistics=statistics, discrepancy=discrepancy_func
    )
    return dist_metric(d_real, d_syn)

def discriminative_metric(Xr, Xs):
    # use LSTM classification model from TSGM zoo.
    model = tsgm.models.zoo["clf_cl_n"](
        seq_len=Xr.shape[1], feat_dim=Xr.shape[2], output_dim=1).model 
    """
        I think at least two classes are needed for ce loss, since clf_cl_n adopts softmax activation.
        If set output_dim=1, all outputs will be 1 after softmax, which is not what we want.
        However, when I set output_dim=2, some errors occur in the following code.
        I have checked the source code of DiscriminativeMetric.
        The error is caused by the following code:
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            if metric is None:
                return sklearn.metrics.accuracy_score(y_test, y_pred)
            else:
                return metric(y_test, y_pred)
        If n_classes is 2, y_pred will be a 2D array, which will cause an error in accuracy_score.
    """
    model.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    )

    discr_metric = tsgm.metrics.DiscriminativeMetric()
    return discr_metric(
            d_hist=Xr, d_syn=Xs, model=model,
            test_size=0.2, random_seed=42, n_epochs=3
        )

class EvaluatorConvLSTM():
    '''
    NB an oversimplified classifier, for educational purposes only.
    '''
    def __init__(self, model):
        self._model = model

    def evaluate(self, D: tsgm.dataset.Dataset, D_test: tsgm.dataset.Dataset) -> float:
        X_train, y_train = D.Xy
        X_test, y_test = D_test.Xy
        
        self._model.fit(X_train, y_train)
        
        y_pred = np.argmax(self._model.predict(X_test), 1)
        y_test = np.argmax(y_test, 1)
        return sklearn.metrics.accuracy_score(y_pred, y_test)



def consistency_metric(d_real, d_syn, Xr, yr):
    seq_len, feat_dim, n_classes = *Xr.shape[1:], yr.shape[-1]
    models = [tsgm.models.zoo["clf_cl_n"](seq_len, feat_dim, n_classes, n_conv_lstm_blocks=i) for i in range(5)]
    for m in models:
        m.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    evaluators = [EvaluatorConvLSTM(m.model) for m in models]
    consistency_metric = tsgm.metrics.ConsistencyMetric(evaluators=evaluators)
    return consistency_metric(d_real, d_syn, d_real)


def downstream_metric(d_real, d_syn, Xr, yr):
    seq_len, feat_dim, n_classes = *Xr.shape[1:], yr.shape[-1]
    downstream_model = tsgm.models.zoo["clf_cl_n"](seq_len, feat_dim, n_classes, n_conv_lstm_blocks=1).model
    downstream_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    evaluator = EvaluatorConvLSTM(downstream_model)

    downstream_perf_metric = tsgm.metrics.DownstreamPerformanceMetric(evaluator)
    return downstream_perf_metric(d_real, d_syn, d_real)


class FlattenTSOneClassSVM:
    def __init__(self, clf):
        self._clf = clf

    def fit(self, X):
        X_fl = X.reshape(X.shape[0], -1)
        self._clf.fit(X_fl)

    def predict(self, X):
        X_fl = X.reshape(X.shape[0], -1)
        return self._clf.predict(X_fl)


def get_dtest(seq_len, feat_dim, model_name):
    if model_name == "tcGAN_Conv4Architecture":
        X_test, y_test = tsgm.utils.gen_sine_const_switch_dataset(100, seq_len, feat_dim, max_value=2, const=1)
    else:
        X_test, y_test = tsgm.utils.gen_sine_vs_const_dataset(100, seq_len, feat_dim, max_value=2, const=1)
        y_test = keras.utils.to_categorical(y_test)
    scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))
    X_test = scaler.fit_transform(X_test).astype(np.float32)
    d_test = tsgm.dataset.Dataset(X_test, y_test)
    return d_test

def privacy_metric(d_real, d_syn, Xr, model_name):
    seq_len, feat_dim, _ = *Xr.shape[1:], 2
    attacker = FlattenTSOneClassSVM(sklearn.svm.OneClassSVM())
    privacy_metric = tsgm.metrics.PrivacyMembershipInferenceMetric(
        attacker=attacker
    )

    d_test = get_dtest(seq_len, feat_dim, model_name)

    # 1 indicates high privacy and 0 -- low privacy.
    return privacy_metric(d_real, d_syn, d_test)

def all_metrics(d_real, d_syn, Xr, yr, model_name):
    metrics = {
        "distance": distance_metric(d_real, d_syn),
        "MMD": MMD_metric(Xr, d_syn.X),
        "discriminative": discriminative_metric(Xr, d_syn.X),
        "consistency": consistency_metric(d_real, d_syn, Xr, yr),
        "downstream": downstream_metric(d_real, d_syn, Xr, yr),
        "privacy": privacy_metric(d_real, d_syn, Xr, model_name)
    }
    print("Metrics for ", model_name)
    for k, v in metrics.items():
        print(f"{k}: {v}")