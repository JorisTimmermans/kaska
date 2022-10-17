#!/usr/bin/env python
"""A general Two layer neural net class

Code by Feng Yin
"""

import numpy as np
from numba import jit
import tensorflow as tf

from tensorflow.keras import layers
from create_training_set import create_training_set, create_validation_set
import pylab as plt

# the forward and backpropogation
# are from https://medium.com/unit8-machine-learning-publication/computing-the-jacobian-matrix-of-a-neural-network-in-python-4f162e5db180
# but added jit for faster speed in the calculation


@jit(nopython=True)
def affine_forward(x, w, b):
    """
    Forward pass of an affine layer
    :param x: input of dimension (D, )
    :param w: weights matrix of dimension (D, M)
    :param b: biais vector of dimension (M, )
    :return output of dimension (M, ), and cache needed for backprop
    """
    out = np.dot(x, w) + b
    cache = (x, w)
    return out, cache


@jit(nopython=True)
def affine_backward(dout, cache):
    """
    Backward pass for an affine layer.
    :param dout: Upstream Jacobian, of shape (O, M)
    :param cache: Tuple of:
      - x: Input data, of shape (D, )
      - w: Weights, of shape (D, M)
    :return the jacobian matrix containing derivatives of the O neural network
            outputs with respect to this layer's inputs, evaluated at x, of
            shape (O, D)
    """
    x, w = cache
    dx = np.dot(dout, w.T)
    return dx


@jit(nopython=True)
def relu_forward(x):
    """ Forward ReLU
    """
    out = np.maximum(np.zeros(x.shape).astype(np.float32), x)
    cache = x
    return out, cache


@jit(nopython=True)
def relu_backward(dout, cache):
    """
    Backward pass of ReLU
    :param dout: Upstream Jacobian
    :param cache: the cached input for this layer
    :return: the jacobian matrix containing derivatives of the O neural
              network outputs with respect tothis layer's inputs,
              evaluated at x.
    """
    x = cache
    dx = dout * np.where(
        x > 0,
        np.ones(x.shape).astype(np.float32),
        np.zeros(x.shape).astype(np.float32),
    )
    return dx


def forward_backward(x, Hidden_Layers, Output_Layers, cal_jac=False):
    layer_to_cache = dict()
    # for each layer, we store the cache needed for backward pass
    [[w1, b1], [w2, b2]] = Hidden_Layers
    a1, cache_a1 = affine_forward(x, w1, b1)
    r1, cache_r1 = relu_forward(a1)
    a2, cache_a2 = affine_forward(r1, w2, b2)
    rets = []
    for output_layer in Output_Layers:
        w3, b3 = output_layer
        r3, cache_r3 = relu_forward(a2)
        out, cache_out = affine_forward(r3, w3, b3)
        if cal_jac:
            dout = affine_backward(np.ones_like(out), cache_out)
            dout = relu_backward(dout, cache_r3)
            dout = affine_backward(dout, cache_a2)
            dout = relu_backward(dout, cache_r1)
            dx = affine_backward(dout, cache_a1)
            ret = [out, dx]
        else:
            ret = out

        rets.append(ret)
    return rets


def training(X, targs, epochs=2000):
    inputs = layers.Input(shape=(X.shape[1],))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = []
    for i in range(targs.shape[1]):
        outputs.append(layers.Dense(1)(x))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False,
    )
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mean_squared_error", "mean_absolute_error"],
    )
    history = model.fit(
        X,
        [targs[:, i] for i in range(targs.shape[1])],
        epochs=epochs,
        batch_size=60,
    )
    return model, history


def relearn(X, targs, model, epochs=2000):
    history = model.fit(
        X,
        [targs[:, i] for i in range(targs.shape[1])],
        epochs=epochs,
        batch_size=60,
    )
    return model, history


def get_layers(model):
    l1 = model.get_layer(index=1).get_weights()
    l2 = model.get_layer(index=2).get_weights()
    Hidden_Layers = [l1, l2]
    Output_Layers = []
    for layer in model.layers[3:]:
        l3 = layer.get_weights()
        Output_Layers.append(l3)

    return Hidden_Layers, Output_Layers


def save_tf_model(model, fname):
    model.save(fname)


def load_tf_Model(fname):
    model = tf.keras.models.load_model(fname)
    return model


def save_np_model(fname, Hidden_Layers, Output_Layers):
    np.savez(fname, Hidden_Layers=Hidden_Layers, Output_Layers=Output_Layers)


def load_np_model(fname):
    f = np.load(fname, allow_pickle=True)
    Hidden_Layers = f.f.Hidden_Layers
    Output_Layers = f.f.Output_Layers
    return Hidden_Layers, Output_Layers


class Two_NN(object):
    def __init__(
        self,
        tf_model=None,
        tf_model_file=None,
        np_model_file=None,
        Hidden_Layers=None,
        Output_Layers=None,
    ):

        if tf_model_file is not None:
            self.tf_model_file = tf_model_file
            self.tf_model = load_tf_Model(self.tf_model_file)
            self.Hidden_Layers, self.Output_Layers = get_layers(self.tf_model)

        if tf_model is not None:
            print('a')
            self.tf_model = tf_model
            self.Hidden_Layers, self.Output_Layers = get_layers(self.tf_model)

        if np_model_file is not None:
            self.np_model_file = np_model_file
            self.Hidden_Layers, self.Output_Layers = load_np_model(
                np_model_file
            )

        if (Hidden_Layers is not None) & (Output_Layers is not None):
            self.Hidden_Layers = Hidden_Layers
            self.Output_Layers = Output_Layers

    def train(
        self,
        X,
        targs,
        iterations=2000,
        tf_fname='model.h5', #("model.json", "model.h5"),
        save_tf_model=False,
    ):
        # self.X, self.targs = X, targs
        # self.iterations = iterations
        if (X is not None) & (targs is not None):
            self.tf_model, self.history = training(X, targs, epochs=iterations)
            self.Hidden_Layers, self.Output_Layers = get_layers(self.tf_model)
            if save_tf_model:
                # self.save_tf_model(self.tf_model, tf_fname)
                self.save_tf_model(tf_fname)
        else:
            raise IOError("X and targs need to have values")

    def relearn(self, X, targs, iterations=2000):
        if hasattr(self, "tf_model"):
            self.tf_model, self.history = relearn(
                X, targs, self.tf_model, epochs=iterations
            )
            self.Hidden_Layers, self.Output_Layers = get_layers(self.tf_model)
        else:
            raise NameError("No tf model to relearn.")

    def predict(self, x, cal_jac=False):
        if hasattr(self, "Hidden_Layers") and hasattr(self, "Output_Layers"):
            x = x.astype(np.float32)
            rets = forward_backward(
                x, self.Hidden_Layers, self.Output_Layers, cal_jac=cal_jac
            )
        else:
            raise NameError(
                "Hidden_Layers and Output_Layers have not yet been defined, " +
                "and please try to train or load a model first."
            )
        return rets

    def save_tf_model(self, fname):
        if hasattr(self, "tf_model"):
            save_tf_model(self.tf_model, fname)
            self.tf_model_file = fname

        else:
            raise NameError("No tf model to save.")

    def save_np_model(self, fname):
        if hasattr(self, "Hidden_Layers") and hasattr(self, "Output_Layers"):
            np.savez(
                fname,
                Hidden_Layers=self.Hidden_Layers,
                Output_Layers=self.Output_Layers,
            )
            self.np_model_file = fname
        else:
            raise NameError(
                "Hidden_Layers and Output_Layers "
                + "have not yet been defined, and please "
                + "try to train or load a model first."
            )


if __name__ == "__main__":
    import numpy as np
    from numba import jit
    import tensorflow as tf
    from tensorflow.keras import layers
    from create_training_set import create_training_set, create_validation_set
    import pylab as plt

    default_dir = "E:/Simulations/Python/MULTIPLY/kaska/kaska/"
    data_dir =default_dir + 'tests/data/'
    inverter_dir = default_dir + 'inverters/'
    from scipy.stats import linregress
    cmd_only = False


    # 1. load emulator
    f = np.load(inverter_dir + "prosail_2NN.npz", allow_pickle=True)
    tnn = Two_NN(Hidden_Layers=f.f.Hidden_Layers, Output_Layers=f.f.Output_Layers)
    tnn.save_np_model('Test_prosail_emulator')

    # 2. load validation dataset (created with real PROSAIL model)
    v = np.load(data_dir + 'vals.npz')
    x = v.f.vals_x
    vals = v.f.vals
    parameters = ['N', 'cab', 'car', 'cb', 'cw', 'cdm', 'lai', 'ala', 'bsoil', 'psoil', 'sza', 'vza', 'raa']

    # 2.1 plot training set of state-variables and reflectances
    # if cmd_only == False:
    #     plt.figure(figsize=[20, 10])
    #     plt.subplot(2, 1, 1)
    #     plt.plot(x.T)
    #     plt.title('Training-set - state variables')
    #     plt.subplot(2, 1, 2)
    #     plt.plot(vals.T)
    #     plt.title('Training-set - reflectances')
    #     plt.show()

    # 3. plot distribution of training-state-variables
    if cmd_only == False:
        # 3.1 plot validation dataset
        fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(16, 16))
        axs = axs.ravel()
        for i in range(13):
            axs[i].hist(x[:, i], bins=100)
        plt.show()


    # N = x[:,0]
    # cab = x[:,1]  # with transformation -> -100 *np.log(xx)   [5 - 80]
    # car = x[:, 2] # with transformation -> -100 *np.log(xx)   [8 - 10]
    # cb = x[:, 3]  #                                           [0 -  1]
    # cw = x[:, 4]  # with transformation -> -1/50 * np.log(xx) [0.005 - 0.03]
    # cw = x[:, 5]  # with transformation -> -1/50 * np.log(xx) [0.004 - 0.007]
    # lai = x[:, 6] # with transformation -> -2*np.log(xx)      [0 - 8]
    # ala = x[:, 7] # with transformation -> xx*90              [45 - 80]
    # bsoil? = x[:, 8]  #reflectivity                           [0.5 - 1.5]
    # psoil? = x[:, 9]  #wetness                                [0-1]
    # sza = x[:, 10]  # with transformation -> np.arccos(xx)    [0 - 60]
    # vza = x[:, 11]  # with transformation -> np.arccos(xx)    [0 - 10]
    # raa = x[:, 12]  # np.rad2deg(np.arccos(x[:,12]))          [0 - 175]

    # xx= x[:,12] #
    # xxt = np.rad2deg(np.arccos(xx))
    #
    # plt.figure(figsize=[10,10])
    # plt.subplot(2, 2, 1)
    # plt.plot(xx)
    # plt.subplot(2, 2, 2)
    # plt.hist(xx, bins=100)
    # plt.subplot(2,2,3)
    # plt.plot(xxt)
    # plt.subplot(2, 2, 4)
    # plt.hist(xxt, bins=100)

    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
    axs = axs.ravel()
    for i in range(9):
        axs[i].hist(vals[:, i], bins=100)
        # plt.ylabel[i]
        # axs[i].title('rho')
    plt.show()


    # 4. Run emulator on input-data from validation-set
    refs = tnn.predict(x)

    # 4.2 evaluate emulator-output against outputs in validation-set for each Band
    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
    axs = axs.ravel()
    for i in range(9):
        axs[i].plot(refs[i], vals[:, i], "o", alpha=0.1)
        # axs[i].set_title(s2a.iloc[100:2100, b_ind[i]+1].name)

        # calculate some basic statistics (using a linear regression analysis)
        lin = linregress(refs[i].ravel(), vals[:, i])
        print(lin.slope, lin.intercept, lin.rvalue, lin.stderr)
    plt.title('evaluate emulator against validation-set')
    plt.show()

    import pdb
    pdb.set_trace()
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    # create training sets
    # N = x[:,0]
    # cab = x[:,1]  # with transformation -> -100 *np.log(xx)   [5 - 80]
    # car = x[:, 2] # with transformation -> -100 *np.log(xx)   [8 - 10]
    # cb = x[:, 3]  #                                           [0 -  1]
    # cw = x[:, 4]  # with transformation -> -1/50 * np.log(xx) [0.005 - 0.03]
    # cw = x[:, 5]  # with transformation -> -1/50 * np.log(xx) [0.004 - 0.007]
    # lai = x[:, 6] # with transformation -> -2*np.log(xx)      [0 - 8]
    # ala = x[:, 7] # with transformation -> xx*90              [45 - 80]
    # bsoil? = x[:, 8]  #reflectivity                           [0.5 - 1.5]
    # psoil? = x[:, 9]  #wetness                                [0-1]
    # sza = x[:, 10]  # with transformation -> np.arccos(xx)    [0 - 60]
    # vza = x[:, 11]  # with transformation -> np.arccos(xx)    [0 - 10]
    # raa = x[:, 12]  # np.rad2deg(np.arccos(x[:,12]))          [0 - 175]

    ntrain = 10000
    parameters = ['N', 'cab', 'car', 'cb', 'cw', 'cdm', 'lai', 'ala', 'bsoil', 'psoil', 'sza', 'vza', 'raa']
    parameters = ['N', 'cab', 'car', 'cb', 'cdm', 'cw', 'lai', 'ala', 'bsoil', 'psoil', 'sza', 'vza', 'raa']
    min_vals   = [1.2,  5,    8,     0,    0.005,0.004, 0,     45,    0.5,     0,       0,      0,     0 ]
    max_vals   = [1.8,  80,   10,    1,    0.030,0.007, 8,     80,    1.5,     1,       60,     10,    175 ]
    training_set_untransformed, distributions = create_training_set(parameters, min_vals, max_vals, n_train=ntrain)
    validate_set = create_validation_set(distributions)

    training_set_transformed = training_set_untransformed*1.
    for i, parameter in enumerate(parameters):
        if parameter == 'cab':
            training_set_transformed[:,i] = np.exp(-0.01 * training_set_untransformed[:,i])
        if parameter == 'car':
            training_set_transformed[:,i] = np.exp(-0.01 * training_set_untransformed[:,i])
        if parameter == 'cw':
            training_set_transformed[:,i] = np.exp(-50*training_set_untransformed[:,i])
        if parameter == 'cdm':
            training_set_transformed[:,i] = np.exp(-50*training_set_untransformed[:,i])
        if parameter == 'lai':
            training_set_transformed[:,i] = np.exp(-0.5*training_set_untransformed[:,i])
        if parameter == 'ala':
            training_set_transformed[:,i] = 1/90. * training_set_untransformed[:,i]
        if parameter == 'sza':
            training_set_transformed[:,i] = np.cos(np.deg2rad(training_set_untransformed[:,i]))
        if parameter == 'vza':
            training_set_transformed[:,i] = np.cos(np.deg2rad(training_set_untransformed[:,i]))
        if parameter == 'raa':
            training_set_transformed[:,i] = np.cos(np.deg2rad(training_set_untransformed[:,i]))

    # 6.Neural networks
    # 6.0 parameters
    band_sel = [1,2,3,4,5,6,7,8]
    param_sel =[0, 1, 3, 6, 7]

    # 6.2 create training set
    xnew = training_set_transformed
    valsnew = tnn.predict(xnew)
    valsnew = np.transpose(np.array(valsnew)[:,:,0])

    training_x = np.hstack((vals[:, band_sel], x[:, 10:]))
    training_y = x[:, param_sel]

    training_x_new = np.hstack((valsnew[:, band_sel], xnew[:, 10:]))
    training_y_new = xnew[:, param_sel]

    # training_x_new
    # band1: 0.009 - 0.287         -0.0127 - 0.343
    # band2: 0.002 - 0.342         -0.1678 - 0.395
    # band3: 0.015 - 0.383         -0.091  - 0.447
    # band4: 0.097 - 0.468         -0.05   - 0.486
    # band5: 0.150 - 0.591          0.013  - 0.595
    # band6: 0.169 - 0.628          0.034  - 0.619
    # band7: 0.061 - 0.633          0.018  - 0.684
    # band8: 0.018 - 0.556         -0.0132 - 0.639

    # ssz:   0.500 - 0.999          0.500  - 1.0
    # vza:   0.984 - 1.000          0.984  - 1.0
    # raa:  -0.999 - 0.999          -0.999 - 1.0

    # 6.3.1 create  neural network with old 'training_set'  and run it
    fname = inverter_dir + 'NNtest.h5'
    tnn.train(training_x, training_y, iterations=2000, tf_fname=fname, save_tf_model=True)
    NN_trained = load_tf_Model(fname)
    trained = np.array(NN_trained.predict(training_x))

    # 6.3.2 create neural network with new 'training_set' and run it
    fname_new = inverter_dir + 'NNtest_new.h5'
    tnn.train(training_x_new, training_y_new, iterations=20, tf_fname=fname_new, save_tf_model=True)
    NN_trained_new = load_tf_Model(fname_new)
    trained_new =np.array(NN_trained_new.predict(training_x))

    # 6.3.3 load original NN  and run it
    fname_orig = inverter_dir + 'Prosail_5_paras.h5'
    NN_loaded = load_tf_Model(fname_orig)
    loaded = np.array(NN_loaded.predict(training_x))

    # 6.4 evaluate new emulators against old one
    E = np.array(trained) - np.array(loaded)
    RMSE = np.sqrt(np.mean(E**2))

    E_new = np.array(trained_new) - np.array(loaded)
    RMSE_new = np.sqrt(np.mean(E_new ** 2))

    # 6.5 evaluate emulators (results) against validation set.
    print(RMSE)
    print(RMSE_new)
    import pdb
    pdb.set_trace()

    np.rad2deg(np.arccos(x[:, 12]))
