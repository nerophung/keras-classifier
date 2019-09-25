from tensorflow.python import keras
from nets.lenet.lenet5 import LeNet5, TrongNet
from keras.applications.vgg16 import VGG16

model = TrongNet((50, 50, 1), 10, "")
model.save("")
