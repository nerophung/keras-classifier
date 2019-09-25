from tensorflow.python import keras


def init_optimizers(optimizer, lr):
    optimizers = {
        'sgd': keras.optimizers.SGD(lr=lr),
        'adadelta': keras.optimizers.Adadelta(lr=lr),
        'adam': keras.optimizers.Adam(lr=lr),
        'adagrad': keras.optimizers.Adagrad(lr=lr),
        'rmsprop': keras.optimizers.RMSprop(lr=lr)
    }
    if optimizer not in optimizers:
        raise ValueError('Optimizer name is not valid.')
    else:
        return optimizers[optimizer]


def init_loss(loss):
    losses = {
        'categorical_crossentropy': keras.losses.categorical_crossentropy,
        'mean_squared_error': keras.losses.mean_squared_error,
        'binary_crossentropy': keras.losses.binary_crossentropy
    }
    if loss not in losses:
        raise ValueError('Loss function name is not valid')
    else:
        return losses[loss]

