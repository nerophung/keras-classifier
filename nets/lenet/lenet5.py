from tensorflow.python.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras import models


def LeNet5(input_shape, num_classes, pretrained_weights=None):
    input_image = layers.Input(shape=input_shape, name='input_1')
    x = layers.Conv2D(filters=6, kernel_size=(3, 3), activation=activations.relu)(input_image)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation=activations.relu)(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation=activations.relu)(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=120, activation=activations.relu)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=84, activation=activations.relu)(x)
    x = layers.Dense(units=num_classes, activation=activations.softmax, name='predictions')(x)
    model = models.Model(input_image, x, name='lenet5')
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


def TrongNet(input_shape, num_classes, pretrained_weights=None):
    input_image = layers.Input(shape=input_shape, name='input_1')
    x = layers.Conv2D(filters=6, kernel_size=(3, 3), activation=activations.relu)(input_image)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation=activations.relu)(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation=activations.relu)(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu)(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=120, activation=activations.relu)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=84, activation=activations.relu)(x)
    x = layers.Dense(units=num_classes, activation=activations.softmax, name='predictions')(x)
    model = models.Model(input_image, x, name='trongnet')
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model
