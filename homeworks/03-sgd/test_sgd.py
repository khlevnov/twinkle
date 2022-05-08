import numpy as np
from numpy.core.numeric import indices
from numpy.random import shuffle
from numpy.testing._private.utils import assert_raises
import pytest
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler as StandardScaler_

from _losses import SquaredLoss
from _sgd import sgd, print_dloss
from MaxAbsScaler import MaxAbsScaler
from MinMaxScaler import MinMaxScaler
from SGDRegressor import SGDRegressor
from StandardScaler import StandardScaler
from metrics import mean_absolute_error, mean_squared_error


def load_synthetic_dataset(intercept=True, n_features=10, n_objects=20, seed=42):
    rng = np.random.default_rng(seed)
    intercept = rng.normal()
    weights = rng.normal(size=n_features)
    X = rng.normal(size=(n_objects, n_features))
    y = X.dot(weights) + intercept
    return X, y


def load_dataset():
    X, y = load_boston(return_X_y=True)
    X = StandardScaler_().fit_transform(X)
    return X, y


def wrap_sgd(sgd):
    X, y = load_dataset()
    rng = np.random.RandomState(42)

    default_args = dict(
        weights=rng.randn(X.shape[1]),
        intercept=np.zeros(1),
        loss=SquaredLoss(),
        X=X,
        y=y,
        max_iter=1000,
        fit_intercept=True,
        verbose=False,
        shuffle=True,
        seed=None,
        eta0=1e-2,
        sample_weight=None,
    )

    def wrappee(**args):
        merged_args = {**default_args, **args}
        return sgd(**merged_args)

    return wrappee


class NdarrayProxy:
    def __init__(self, wrappee):
        self.__wrappee = wrappee
        self.order = []

    def __getattr__(self, attr):
        return getattr(self.__wrappee, attr)

    def __getitem__(self, i):
        if len(self.order) == 0:
            self.order.append(i)
        elif i != self.order[-1]:
            self.order.append(i)
        return self.__wrappee[i]

    def __len__(self):
        return len(self.__wrappee)


class TestSquaredLoss():
    def test_loss(self):
        loss = SquaredLoss()
        assert loss.loss(0, 0) == 0
        assert loss.loss(1, 1) == 0
        assert loss.loss(2, 2) == 0
        assert loss.loss(0, 1) == 0.5
        assert loss.loss(0, 2) == 2
        assert loss.loss(1, 0) == 0.5
        assert loss.loss(2, 0) == 2

    def test_dloss(self):
        loss = SquaredLoss()
        assert loss.dloss(0, 0) == 0
        assert loss.dloss(1, 1) == 0
        assert loss.dloss(2, 2) == 0
        assert loss.dloss(0, 1) == -1
        assert loss.dloss(0, 2) == -2
        assert loss.dloss(1, 0) == 1
        assert loss.dloss(2, 0) == 2


class TestSgdFn():
    def test_return_epochs_with_no_early_stopping(self):
        _sgd = wrap_sgd(sgd)
        _, _, epochs = _sgd(max_iter=1)
        assert epochs == 1
        _, _, epochs = _sgd(max_iter=10)
        assert epochs == 10
        _, _, epochs = _sgd(max_iter=100)
        assert epochs == 100

    def test_fit_only_weights(self):
        _sgd = wrap_sgd(sgd)
        weights, intercept, _ = _sgd(
            intercept=np.array([0]),
            fit_intercept=False,
            shuffle=False,
        )
        assert np.allclose(weights, np.array([
            -1.10, -4.82, 3.81, 5.40, 7.04, -2.39, 4.12,
            -6.84, -0.67,  2.03, 3.51, 1.51, -7.37,
        ]), atol=5e-2)
        assert np.allclose(intercept, np.array([0]))

    def test_fit_weights_fit_intercept(self):
        _sgd = wrap_sgd(sgd)
        weights, intercept, _ = _sgd(shuffle=False)
        assert np.allclose(weights, np.array([
            -1.16, 0.65, 0.21, 1.31, -1.96, 0.71, -0.23,
            -2.13, 1.97, -1.41, -1.37, 0.27, -4.48,
        ]), atol=5e-2)
        assert np.allclose(intercept, np.array([21.75]), atol=5e-2)

    def test_fit_given_weights(self):
        _sgd = wrap_sgd(sgd)
        initial_weights = np.ones(13)
        initial_intercept = np.zeros(1)
        _sgd(
            weights=initial_weights,
            intercept=initial_intercept,
            shuffle=False,
        )
        assert np.allclose(initial_weights, np.array([
            -1.16, 0.65, 0.21, 1.31, -1.96, 0.71, -0.23,
            -2.13, 1.97, -1.41, -1.37, 0.27, -4.48,
        ]), atol=5e-2)
        assert np.allclose(initial_intercept, np.array([21.75]), atol=5e-2)


class TestSGDRegressor():
    def test_validate_loss(self):
        reg = SGDRegressor(loss='squared_loss')
        with assert_raises(ValueError):
            reg = SGDRegressor(loss='unknown_loss')

    def test_validate_max_iter(self):
        reg = SGDRegressor(max_iter=10)
        with assert_raises(ValueError):
            reg = SGDRegressor(max_iter=-10)
        with assert_raises(ValueError):
            reg = SGDRegressor(max_iter=0)

    def test_coef_attribute(self):
        X, y = np.zeros((1, 13)), np.zeros(1)
        reg = SGDRegressor(max_iter=1)

        with assert_raises(AttributeError):
            reg.coef_

        reg.fit(X, y)
        assert reg.coef_.shape == (13,)

    def test_intercept_attribute(self):
        X, y = np.zeros((1, 13)), np.zeros(1)
        reg = SGDRegressor(max_iter=1)

        with assert_raises(AttributeError):
            reg.intercept_

        reg.fit(X, y)
        assert reg.intercept_.shape == (1,)

    def test_fit_only_weights(self):
        reg = SGDRegressor(fit_intercept=False, shuffle=False)\
            .fit(*load_dataset())
        assert np.allclose(reg.intercept_, np.array([0]))
        assert np.allclose(reg.coef_, np.array([
            -1.10, -4.82, 3.81, 5.40, 7.04, -2.39, 4.12,
            -6.84, -0.67,  2.03, 3.51, 1.51, -7.37,
        ]), atol=5e-2)

    def test_fit_weights_fit_intercept(self):
        reg = SGDRegressor(shuffle=False)\
            .fit(*load_dataset())
        assert np.allclose(reg.coef_, np.array([
            -1.16, 0.65, 0.21, 1.31, -1.96, 0.71, -0.23,
            -2.13, 1.97, -1.41, -1.37, 0.27, -4.48,
        ]), atol=5e-2)
        assert np.allclose(reg.intercept_, np.array([21.75]), atol=5e-2)

    def test_predict(self):
        pytest.skip('not implemented')


class TestPrintDloss():
    def test_no_verbose(self, capfd):
        print_dloss(42.0, False)
        out, _ = capfd.readouterr()
        assert out == ''

    def test_nan(self, capfd):
        print_dloss(np.nan)
        out, _ = capfd.readouterr()
        assert out == ''

    def test_zero(self, capfd):
        print_dloss(0.0)
        out, _ = capfd.readouterr()
        assert out == '-- grad +0.00e+00\n'

    def test_positive1(self, capfd):
        print_dloss(0.001)
        out, _ = capfd.readouterr()
        assert out == '-- grad +1.00e-03\n'

    def test_positive2(self, capfd):
        print_dloss(0.01)
        out, _ = capfd.readouterr()
        assert out == '-- grad +1.00e-02\n'

    def test_positive3(self, capfd):
        print_dloss(10)
        out, _ = capfd.readouterr()
        assert out == '-- grad +1.00e+01\n'

    def test_negative(self, capfd):
        print_dloss(-123.123)
        out, _ = capfd.readouterr()
        assert out == '-- grad -1.23e+02\n'

    def test_inside_sgd_no_verbose(self, capfd):
        pytest.skip('not implemented')


class TestMaxAbsScaler():
    def test_fit_chainable(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer

    def test_fit_n_samples_seen(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4

    def test_fit_max_abs_easy(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.max_abs_, np.array([ 1., 10., 10., 100.]))

    def test_fit_max_abs_from_docs(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1., -1.,  2.],
            [ 2.,  0.,  0.],
            [ 0.,  1., -1.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 3
        assert np.allclose(transformer.max_abs_, np.array([ 2., 1., 2.]))

    def test_fit_scale_easy(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.max_abs_, np.array([ 1., 10., 10., 100.]))
        assert np.allclose(transformer.scale_, np.array([ 1., 10., 10., 100.]))

    def test_fit_scale_from_docs(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1., -1.,  2.],
            [ 2.,  0.,  0.],
            [ 0.,  1., -1.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 3
        assert np.allclose(transformer.max_abs_, np.array([ 2., 1., 2.]))
        assert np.allclose(transformer.scale_, np.array([ 2., 1., 2.]))

    def test_transform_easy(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        transformer.fit(X)
        X_scaled = transformer.transform(X)
        assert np.allclose(X_scaled, np.array([
            [ 1.,  0.7, -0.2, 0.01],
            [ 1., -1.0,  1.0,  1.0],
            [ 1.,  0.5, -0.7, 0.02],
            [ 1.,  0.2, -0.5, 0.42],
        ]))

    def test_transform_from_docs(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1., -1.,  2.],
            [ 2.,  0.,  0.],
            [ 0.,  1., -1.],
        ])
        transformer.fit(X)
        X_scaled = transformer.transform(X)
        assert np.allclose(X_scaled, np.array([
            [ 0.5, -1.0,  1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0, -0.5],
        ]))

    def test_fit_transform_easy(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        X_scaled = transformer.fit_transform(X)
        assert np.allclose(transformer.scale_, np.array([ 1., 10., 10., 100.]))
        assert np.allclose(transformer.max_abs_, np.array([ 1., 10., 10., 100.]))
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(X_scaled, np.array([
            [ 1.,  0.7, -0.2, 0.01],
            [ 1., -1.0,  1.0,  1.0],
            [ 1.,  0.5, -0.7, 0.02],
            [ 1.,  0.2, -0.5, 0.42],
        ]))

    def test_fit_transform_from_docs(self):
        transformer = MaxAbsScaler()
        X = np.array([
            [ 1., -1.,  2.],
            [ 2.,  0.,  0.],
            [ 0.,  1., -1.],
        ])
        X_scaled = transformer.fit_transform(X)
        assert np.allclose(transformer.scale_, np.array([ 2., 1., 2.]))
        assert np.allclose(transformer.max_abs_, np.array([ 2., 1., 2.]))
        assert transformer.n_samples_seen_ == 3
        assert np.allclose(X_scaled, np.array([
            [ 0.5, -1.0,  1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0, -0.5],
        ]))


class TestMinMaxScaler():
    def test_fit_chainable(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer

    def test_fit_n_samples_seen(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4

    def test_fit_data_min(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.data_min_, np.array([ 1., -10.,  -7.,   1.]))

    def test_fit_data_max(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.data_min_, np.array([ 1., -10.,  -7.,   1.]))
        assert np.allclose(transformer.data_max_, np.array([ 1.,   7.,  10., 100.]))

    def test_fit_data_range(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.data_min_, np.array([ 1., -10.,  -7.,   1.]))
        assert np.allclose(transformer.data_max_, np.array([ 1.,   7.,  10., 100.]))
        assert np.allclose(transformer.data_range_, np.array([ 0., 17., 17.,  99.]))

    def test_fit_scale(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.data_min_, np.array([ 1., -10.,  -7.,   1.]))
        assert np.allclose(transformer.data_max_, np.array([ 1.,   7.,  10., 100.]))
        assert np.allclose(transformer.data_range_, np.array([ 0., 17., 17.,  99.]))
        assert np.allclose(transformer.scale_, np.array([ 1., 0.058, 0.058, 0.01]), atol=1e-2)

    def test_fit_min(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.data_min_, np.array([ 1., -10.,  -7.,   1.]))
        assert np.allclose(transformer.data_max_, np.array([ 1.,   7.,  10., 100.]))
        assert np.allclose(transformer.data_range_, np.array([ 0., 17., 17.,  99.]))
        assert np.allclose(transformer.scale_, np.array([ 1., 0.058, 0.058, 0.01]), atol=1e-2)
        assert np.allclose(transformer.min_, np.array([ -1., 0.58, 0.41, -0.01]), atol=1e-2)

    def test_fit_in_feature_range(self):
        transformer = MinMaxScaler(feature_range=(2, 4))
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.data_min_, np.array([ 1., -10.,  -7.,   1.]))
        assert np.allclose(transformer.data_max_, np.array([ 1.,   7.,  10., 100.]))
        assert np.allclose(transformer.data_range_, np.array([ 0., 17., 17.,  99.]))
        assert np.allclose(transformer.scale_, np.array([ 2., 0.11, 0.11, 0.02]), atol=1e-2)
        assert np.allclose(transformer.min_, np.array([ 0., 3.17, 2.82, 1.97]), atol=1e-2)

    def test_transform(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        X_scaled = transformer.transform(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.data_min_, np.array([ 1., -10.,  -7.,   1.]))
        assert np.allclose(transformer.data_max_, np.array([ 1.,   7.,  10., 100.]))
        assert np.allclose(transformer.data_range_, np.array([ 0., 17., 17.,  99.]))
        assert np.allclose(transformer.scale_, np.array([ 1., 0.058, 0.058, 0.01]), atol=1e-2)
        assert np.allclose(transformer.min_, np.array([ -1., 0.58, 0.41, -0.01]), atol=1e-2)
        assert np.allclose(X_scaled, np.array([
            [ 0.,    1., 0.29,     0.],
            [ 0.,    0.,    1.,    1.],
            [ 0., 0.88,     0., 0.01 ],
            [ 0., 0.70,  0.11,  0.41 ],
        ]), atol=1e-2)

    def test_fit_transform(self):
        transformer = MinMaxScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        X_scaled = transformer.fit_transform(X)
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.data_min_, np.array([ 1., -10.,  -7.,   1.]))
        assert np.allclose(transformer.data_max_, np.array([ 1.,   7.,  10., 100.]))
        assert np.allclose(transformer.data_range_, np.array([ 0., 17., 17.,  99.]))
        assert np.allclose(transformer.scale_, np.array([ 1., 0.058, 0.058, 0.01]), atol=1e-2)
        assert np.allclose(transformer.min_, np.array([ -1., 0.58, 0.41, -0.01]), atol=1e-2)
        assert np.allclose(X_scaled, np.array([
            [ 0.,    1., 0.29,     0.],
            [ 0.,    0.,    1.,    1.],
            [ 0., 0.88,     0., 0.01 ],
            [ 0., 0.70,  0.11,  0.41 ],
        ]), atol=1e-2)


class TestStandardScaler():
    def test_fit_chainable(self):
        transformer = StandardScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer

    def test_fit_n_samples_seen(self):
        transformer = StandardScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4

    def test_fit_mean(self):
        transformer = StandardScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.mean_, np.array([1., 1., -1., 36.25]), atol=1e-2)

    def test_fit_var(self):
        transformer = StandardScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.mean_, np.array([1., 1., -1., 36.25]), atol=1e-2)
        assert np.allclose(transformer.var_, np.array([0., 43.5, 43.5, 1628.1875]), atol=1e-2)

    def test_fit_var_without_std(self):
        transformer = StandardScaler(with_std=False)
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.mean_, np.array([1., 1., -1., 36.25]), atol=1e-2)
        assert transformer.var_ is None

    def test_fit_mean_var_without_mean_std(self):
        transformer = StandardScaler(with_mean=False, with_std=False)
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert transformer.mean_ is None
        assert transformer.var_ is None

    def test_fit_scale(self):
        transformer = StandardScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.var_, np.array([0., 43.5, 43.5, 1628.1875]), atol=1e-2)
        assert np.allclose(transformer.mean_, np.array([1., 1., -1., 36.25]), atol=1e-2)
        assert np.allclose(transformer.scale_, np.array([1., 6.59, 6.59, 40.35]), atol=1e-2)

    def test_fit_scale_without_std(self):
        transformer = StandardScaler(with_std=False)
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.mean_, np.array([1., 1., -1., 36.25]), atol=1e-2)
        assert transformer.var_ is None
        assert transformer.scale_ is None

    def test_fit_scale_without_mean_std(self):
        transformer = StandardScaler(with_mean=False, with_std=False)
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert transformer.var_ is None
        assert transformer.mean_ is None
        assert transformer.scale_ is None

    def test_transform(self):
        transformer = StandardScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.var_, np.array([0., 43.5, 43.5, 1628.1875]), atol=1e-2)
        assert np.allclose(transformer.mean_, np.array([1., 1., -1., 36.25]), atol=1e-2)
        assert np.allclose(transformer.scale_, np.array([1., 6.59, 6.59, 40.35]), atol=1e-2)
        X_scaled = transformer.transform(X)
        assert np.allclose(X_scaled, np.array([
            [ 0.,  0.90, -0.15, -0.87],
            [ 0., -1.66,  1.66,  1.57],
            [ 0.,  0.60, -0.90, -0.84],
            [ 0.,  0.15, -0.60,  0.14],
        ]), atol=1e-2)

    def test_transform_without_std(self):
        transformer = StandardScaler(with_std=False)
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.mean_, np.array([1., 1., -1., 36.25]), atol=1e-2)
        assert transformer.var_ is None
        assert transformer.scale_ is None
        X_scaled = transformer.transform(X)
        assert np.allclose(X_scaled, np.array([
            [ 0.,   6.,  -1., -35.25],
            [ 0., -11.,  11.,  63.75],
            [ 0.,   4.,  -6., -34.25],
            [ 0.,   1.,  -4.,   5.75],
        ]), atol=1e-2)

    def test_transform_without_mean_std(self):
        transformer = StandardScaler(with_mean=False, with_std=False)
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        fitted_transformer = transformer.fit(X)
        assert transformer == fitted_transformer
        assert transformer.n_samples_seen_ == 4
        assert transformer.var_ is None
        assert transformer.mean_ is None
        assert transformer.scale_ is None
        X_scaled = transformer.transform(X)
        assert np.allclose(X_scaled, np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ]), atol=1e-2)

    def test_fit_transform(self):
        transformer = StandardScaler()
        X = np.array([
            [ 1.,   7., -2.,   1.],
            [ 1., -10., 10., 100.],
            [ 1.,   5., -7.,   2.],
            [ 1.,   2., -5.,  42.],
        ])
        X_scaled = transformer.fit_transform(X)
        assert transformer.n_samples_seen_ == 4
        assert np.allclose(transformer.var_, np.array([0., 43.5, 43.5, 1628.1875]), atol=1e-2)
        assert np.allclose(transformer.mean_, np.array([1., 1., -1., 36.25]), atol=1e-2)
        assert np.allclose(transformer.scale_, np.array([1., 6.59, 6.59, 40.35]), atol=1e-2)
        assert np.allclose(X_scaled, np.array([
            [ 0.,  0.90, -0.15, -0.87],
            [ 0., -1.66,  1.66,  1.57],
            [ 0.,  0.60, -0.90, -0.84],
            [ 0.,  0.15, -0.60,  0.14],
        ]), atol=1e-2)


class TestMetrics:
    def test_absolute_easy(self):
        assert mean_absolute_error(np.array([0]), np.array([0])) == 0
        assert mean_absolute_error(np.array([0]), np.array([42])) == 42
        assert mean_absolute_error(np.array([42]), np.array([0])) == 42
        assert mean_absolute_error(np.array([42]), np.array([42])) == 0
        assert mean_absolute_error(np.zeros((42,)), np.zeros((42,))) == 0
        assert mean_absolute_error(np.zeros((42,)), np.ones((42,))) == 1
        assert mean_absolute_error(np.ones((42,)), np.zeros((42,))) == 1
        assert mean_absolute_error(np.ones((42,)), np.ones((42,))) == 0

    def test_squared_easy(self):
        assert mean_squared_error(np.array([0]), np.array([0])) == 0
        assert mean_squared_error(np.array([0]), np.array([42])) == 1764.0
        assert mean_squared_error(np.array([42]), np.array([0])) == 1764.0
        assert mean_squared_error(np.array([42]), np.array([42])) == 0
        assert mean_squared_error(np.zeros((42,)), np.zeros((42,))) == 0
        assert mean_squared_error(np.zeros((42,)), np.ones((42,))) == 1
        assert mean_squared_error(np.ones((42,)), np.zeros((42,))) == 1
        assert mean_squared_error(np.ones((42,)), np.ones((42,))) == 0


class TestShuffle:
    def test_sgd_every_item(self):
        X, _ = load_dataset()

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1, shuffle=False)
        assert len(X_proxy.order) == len(X)

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1, seed=42)
        assert len(X_proxy.order) == len(X)

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1)
        assert len(X_proxy.order) == len(X)

    def test_sgd_no_shuffle(self):
        X, _ = load_dataset()

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1, shuffle=False)
        assert np.allclose(X_proxy.order, list(range(len(X))))

    def test_sgd_with_given_seed(self):
        X, _ = load_dataset()

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1, seed=42)
        prev_order = X_proxy.order

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1, seed=42)
        assert np.allclose(X_proxy.order, prev_order)

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1, seed=17)
        assert not np.allclose(X_proxy.order, prev_order)

    def test_sgd_with_empty_seed(self):
        X, _ = load_dataset()

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1)
        prev_order = X_proxy.order

        X_proxy = NdarrayProxy(X)
        wrap_sgd(sgd)(X=X_proxy, max_iter=1)
        assert not np.allclose(X_proxy.order, prev_order)

    def test_reg_no_shuffle(self):
        X, y = load_dataset()

        X_proxy = NdarrayProxy(X)
        SGDRegressor(max_iter=1, shuffle=False).fit(X_proxy, y)
        assert X_proxy.order == list(range(len(X)))

    def test_reg_with_given_seed(self):
        X, y = load_dataset()

        X_proxy = NdarrayProxy(X)
        SGDRegressor(max_iter=1, random_state=42).fit(X_proxy, y)
        prev_order = X_proxy.order

        X_proxy = NdarrayProxy(X)
        SGDRegressor(max_iter=1, random_state=42).fit(X_proxy, y)
        assert np.allclose(X_proxy.order, prev_order)

        X_proxy = NdarrayProxy(X)
        SGDRegressor(max_iter=1, random_state=17).fit(X_proxy, y)
        assert not np.allclose(X_proxy.order, prev_order)

    def test_reg_with_empty_seed(self):
        X, y = load_dataset()

        X_proxy = NdarrayProxy(X)
        SGDRegressor(max_iter=1).fit(X_proxy, y)
        prev_order = X_proxy.order

        X_proxy = NdarrayProxy(X)
        SGDRegressor(max_iter=1).fit(X_proxy, y)
        assert not np.allclose(X_proxy.order, prev_order)


class TestPartialFit:
    def test_chainable(self):
        X, y = load_dataset()
        reg = SGDRegressor(max_iter=1)
        maybe_another_reg = reg.partial_fit(X, y)

        assert maybe_another_reg == reg

    def test_coef_shape(self):
        X, y = load_dataset()
        reg = SGDRegressor(max_iter=1)

        with assert_raises(AttributeError):
            reg.coef_

        reg.partial_fit(X, y)
        assert reg.coef_.shape == (13,)

    def test_intercept_shape(self):
        X, y = load_dataset()
        reg = SGDRegressor(max_iter=1)

        with assert_raises(AttributeError):
            reg.intercept_

        reg.partial_fit(X, y)
        assert reg.intercept_.shape == (1,)

    def test_runs_one_iter_only(self):
        X, y = load_dataset()
        X_proxy = NdarrayProxy(X)
        SGDRegressor(max_iter=1).fit(X_proxy, y)
        assert len(X_proxy.order) == len(X)

    def test_use_fitted_weights(self):
        X, y = load_dataset()
        reg = SGDRegressor(fit_intercept=False, max_iter=999, shuffle=False).fit(X, y)
        for _ in range(1):
            reg.partial_fit(X, y)

        assert np.allclose(reg.intercept_, np.array([0]))
        assert np.allclose(reg.coef_, np.array([
            -1.10, -4.82, 3.81, 5.40, 7.04, -2.39, 4.12,
            -6.84, -0.67,  2.03, 3.51, 1.51, -7.37,
        ]), atol=5e-2)

    def test_use_fitted_intercept(self):
        X, y = load_dataset()
        reg = SGDRegressor(max_iter=999, shuffle=False).fit(X, y)
        for _ in range(1):
            reg.partial_fit(X, y)

        assert np.allclose(reg.coef_, np.array([
            -1.16, 0.65, 0.21, 1.31, -1.96, 0.71, -0.23,
            -2.13, 1.97, -1.41, -1.37, 0.27, -4.48,
        ]), atol=5e-2)
        assert np.allclose(reg.intercept_, np.array([21.75]), atol=5e-2)


class TestWarmStart:
    def test_disabled(self):
        pytest.skip('not implemented')


class TestSampleWeights:
    def test_no_weights(self):
        pytest.skip('not implemented')


class TestR2Score:
    def test_r2_score(self):
        pytest.skip('not implemented')
