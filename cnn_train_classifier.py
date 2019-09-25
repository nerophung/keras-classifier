from __future__ import print_function
from __future__ import division

import tensorflow as tf
from lib.model_utils import init_loss, init_optimizers
from nets.lenet.lenet5 import LeNet5, TrongNet
from tensorflow.python import keras
import os
import glob
import numpy as np
import tqdm
tf.random.set_random_seed(101)
np.random.seed(101)

# Directory PATH
tf.app.flags.DEFINE_string('dataset_dir', '', 'Directory where store tf_record files.')
tf.app.flags.DEFINE_string('output_dir', '', 'Directory where srote checkpoints.')
# Data input and output
tf.app.flags.DEFINE_integer('image_size', 100, 'Size of image.')
tf.app.flags.DEFINE_string('split_train_name', 'train-*.tfrecord', 'Name for split dataset.')
tf.app.flags.DEFINE_string('split_validation_name', 'validation-*.tfrecord', 'Name for split dataset.')
tf.app.flags.DEFINE_string('output_file_name', 'final_models.hdf5', 'Name of output file.')
tf.app.flags.DEFINE_string('checkpoint_pattern', 'weights-{epoch:02d}-{acc:.2f}.hdf5',
                           'Checkpoint pattern for saving')
# Training configurations
tf.app.flags.DEFINE_float('validation_split', None, 'Ratio of validation data.')
tf.app.flags.DEFINE_string('pretrained_weights', None, 'Path to pretrained weights file (hdf5)')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch size.')
tf.app.flags.DEFINE_integer('num_classes', 3581, 'Number of classes.')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs.')
tf.app.flags.DEFINE_bool('using_augmentation', False, 'Using Image Generator from Keras')
# Optimizer
tf.app.flags.DEFINE_string('optimizer', 'adadelta', 'Optimizer algorithm')
tf.app.flags.DEFINE_float('lr', 0.001, 'Amount of Learning rate')
tf.app.flags.DEFINE_string('loss', 'categorical_crossentropy', 'Loss function')
# Tensor board
tf.app.flags.DEFINE_string('monitor_checkpoint', 'acc', 'Value that have been checked for save weights')
tf.app.flags.DEFINE_string('monitor_mode', 'max', 'Mode for compare old state')

FLAGS = tf.app.flags.FLAGS


def create_dataset(file_path):
    images = []
    labels = []
    features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64)
                }
    for s_example in tf.python_io.tf_record_iterator(file_path):
        example = tf.parse_single_example(s_example, features=features)
        image = tf.image.decode_png(example['image/encoded'])
        images.append(image)
        labels.append(tf.one_hot(example['image/class/label'], FLAGS.num_classes))
    return images, labels


def load_data(dataset_dir, pattern):
    dir_pattern = os.path.join(dataset_dir, pattern)
    list_files = glob.glob(dir_pattern, recursive=True)
    list_files.sort()
    if len(list_files) == 0:
        raise ValueError('There is not any tf record file in {} folder'.format(os.path.basename(dataset_dir)))
    images = []
    labels = []
    with tqdm.tqdm(total=len(list_files)) as pbar:
        for file in list_files:
            image, label = create_dataset(os.path.join(dataset_dir, file))
            images.extend(image)
            labels.extend(label)
            pbar.update(1)
    return images, labels


def get_data_augment():
    data_augment = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.2
    )
    return data_augment


def main(_):
    # Log configurations of FLAGS
    model = TrongNet((50, 50, 1), FLAGS.num_classes, FLAGS.pretrained_weights)
    log_file = os.path.join(FLAGS.output_dir, 'config.txt')
    if not os.path.isdir(FLAGS.dataset_dir):
        raise FileExistsError('Directory {} is not exits'.format(FLAGS.dataset_dir))
    with open(log_file, 'w') as log:
        for key, value in FLAGS.flag_values_dict().items():
            log.writelines("{} : {}\n".format(key, value))
        with tf.Session() as sess:
            print('Load data training:')
            training_data = load_data(FLAGS.dataset_dir, FLAGS.split_train_name)
            images_train, labels_train = sess.run(training_data)
            images_train = np.array(images_train)
            labels_train = np.array(labels_train)
            print('Total records for training: {}'.format(images_train.shape[0]))
            log.writelines('Training data: {}\n'.format(images_train.shape[0]))
            print('Loading data validating...')
            validating_data = load_data(FLAGS.dataset_dir, FLAGS.split_validation_name)
            images_val, labels_val = sess.run(validating_data)
            labels_val = np.array(labels_val)
            images_val = np.array(images_val)
            print('Total records for validating: {}'.format(images_val.shape[0]))
            log.writelines('Validating data: {}\n'.format(images_val.shape[0]))
            model.summary(print_fn=lambda x: log.write(x + '\n'))
    optimizer = None
    loss = None
    try:
        # Get optimizer
        optimizer = init_optimizers(FLAGS.optimizer, FLAGS.lr)
        loss = init_loss(FLAGS.loss)
    except ValueError:
        pass

    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(FLAGS.output_dir, FLAGS.checkpoint_pattern),
                                                 monitor=FLAGS.monitor_checkpoint, verbose=1,
                                                 save_best_only=True, mode=FLAGS.monitor_mode)
    tensor_board = keras.callbacks.TensorBoard(log_dir=os.path.join(FLAGS.output_dir, 'tensor_board'), write_graph=True,
                                               write_images=True)
    callbacks_list = [checkpoint, tensor_board]
    model.compile(loss=loss,
                  optimizer=optimizer, metrics=['accuracy'])

    if FLAGS.using_augmentation:
        data_augment = get_data_augment()
        data_augment.fit(images_train)
        model.fit_generator(data_augment.flow(images_train, labels_train, batch_size=FLAGS.batch_size),
                            steps_per_epoch=len(images_train) // FLAGS.batch_size, epochs=FLAGS.epochs,
                            verbose=1, callbacks=callbacks_list, validation_data=(images_val, labels_val),
                            validation_steps=len(images_val) // FLAGS.batch_size)
    else:
        model.fit(images_train, labels_train, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
                  callbacks=callbacks_list, validation_split=FLAGS.validation_split,
                  validating_data=(images_val, labels_val), verbose=1)

    model.save(os.path.join(FLAGS.output_dir, FLAGS.output_file_name))
    print('Model is saved!')


if __name__ == '__main__':
    tf.app.run()
