from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random
import os
import cv2
import tensorflow as tf
import math
import imghdr
import numpy as np
from tqdm import tqdm


IMAGE_FORMAT_DICT = {
    'jpeg': b'jpg',
    'png': b'png',
}


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def separate_train_val_data(image_class_dirs, ratio_train_data=1.0):
    """ Separate image paths in image class directories to training and valiation sets

    :param image_class_dirs: a list of image class directory paths
    :param ratio_train_data: a value between 0 and 1 decides the portion of images used for training data
    :return: a tuple (train, val) with train, val are lists and each contains lists of image paths for each class
             correspondingly.
    """
    train = []
    val = []
    img_extensions = ['jpg', 'png']
    for class_dir_path in image_class_dirs:
        image_paths = []
        for extension in img_extensions:
            image_paths.extend(glob.glob(class_dir_path + '/*.' + extension))

        total_images = len(image_paths)
        num_train_images = int(total_images * ratio_train_data)
        random.shuffle(image_paths)

        train.append(image_paths[:num_train_images])
        val.append(image_paths[num_train_images:])

    return train, val


def write_image_record(writer, image_path, image_label):
    # read and get image information
    image_raw = tf.gfile.FastGFile(image_path, 'rb').read()
    image_array = cv2.imread(image_path)

    image_height, image_width, _ = image_array.shape

    # Get image format
    image_format = IMAGE_FORMAT_DICT.get(imghdr.what(image_path))

    # write image data to file
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_raw),
        'image/format': _bytes_feature(image_format),
        'image/class/label': _int64_feature(image_label),
        'image/height': _int64_feature(image_height),
        'image/width': _int64_feature(image_width),
    }))

    writer.write(example.SerializeToString())


def prepare_data_in_tfrecord_format(train, output_dir, split_name='train',
                                    output_file_prefix='data', ratio_examples_each_class=1, duplicate=True,
                                    verbose=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    num_collected_examples_each_class = [int(len(image_paths) * ratio_examples_each_class) for image_paths in train]
    num_examples_each_class = max(num_collected_examples_each_class)

    if duplicate:
        tfrecord_file_num = len(train)
    else:
        tfrecord_file_num = math.ceil(sum(num_collected_examples_each_class) / num_examples_each_class)

    file_index = 0
    num_examples = 0
    writer = None
    with open(os.path.join(output_dir, '{}_info.txt'.format(split_name)), 'w') as info_file:
        with open(os.path.join(output_dir, '{}_file_names.txt'.format(split_name)), 'w') as name_file:
            for example_index in range(num_examples_each_class):
                for class_index in range(len(train)):
                    # open new tfrecord file to write if there is no opening file
                    if not writer:
                        if output_file_prefix:
                            prefix_name = '{}_'.format(output_file_prefix)
                        else:
                            prefix_name = ''
                        output_file_path = os.path.join(output_dir,
                                                        '{}{}-{:05d}-of-{:05d}.tfrecord'.format(prefix_name, split_name,
                                                                                                file_index,
                                                                                                tfrecord_file_num))
                        writer = tf.python_io.TFRecordWriter(output_file_path)

                    # write current image record to file
                    image_path = None
                    if duplicate:
                        image_path = train[class_index][example_index % len(train[class_index])]
                    else:
                        if example_index < num_collected_examples_each_class[class_index]:
                            image_path = train[class_index][example_index]
                        else:
                            continue

                    image_file_name = os.path.basename(image_path)
                    name_file.write("%s\n" % image_file_name)

                    if verbose:
                        print("Adding '%s' to '%s' ..." %
                              (image_path,
                               '{}-{:05d}-of-{:05d}.tfrecord'.format(split_name, file_index, tfrecord_file_num)),
                              end='')
                    write_image_record(writer, image_path=image_path, image_label=class_index)
                    if verbose:
                        print("Done")

                    # update counter for number of examples in current opening file,
                    # if number of examples in current file is equal to max number of examples in a file, close current file.
                    num_examples += 1
                    if num_examples == num_examples_each_class:
                        info_file.write("#examples in %s: %d\n" %
                                        ('{}-{:05d}-of-{:05d}.tfrecord'.format(split_name, file_index,
                                                                               tfrecord_file_num), num_examples))
                        writer.close()
                        writer = None
                        num_examples = 0
                        file_index += 1

        if num_examples > 0:
            info_file.write("#examples in %s: %d\n" %
                            ('{}-{:05d}-of-{:05d}.tfrecord'.format(split_name, file_index, tfrecord_file_num),
                             num_examples))


def convert_dataset_to_tfrecords(dataset_dir, data_dir,
                                 ratio_train_data=1, output_file_prefix='', duplicate_training_data=True,
                                 use_slim_framework=True, verbose=True, num_tf_file=10, num_threads=3, version=1):
    print("Started converting data from '%s' directory to tfrecord format" % dataset_dir)

    class_names = sorted(os.listdir(dataset_dir))
    image_class_dirs = [os.path.join(dataset_dir, name) for name in class_names]
    num_classes = len(class_names)
    print("There are %d classes: %s" % (num_classes, str(class_names)))
    # Prepare lists of image paths for training set and valuation set
    print("Separating data to training set and validation set ...", end='')
    train, val = separate_train_val_data(image_class_dirs, ratio_train_data=ratio_train_data)
    print("Done")

    # Prepare tfrecord files for training data
    print("Preparing training set in tfrecord format ...", end='')
    if use_slim_framework:
        train_data_dir = data_dir
    else:
        train_data_dir = os.path.join(data_dir, 'train')
    prepare_data_in_tfrecord_format(train, train_data_dir, split_name='train', output_file_prefix=output_file_prefix,
                             ratio_examples_each_class=1, duplicate=duplicate_training_data, verbose=verbose,
                             num_tf_file=num_tf_file, num_threads=num_threads, version=version)
    print("Done")

    # Prepare tfrecord files for validation data
    print("Preparing validation set in tfrecord format ...", end='')
    if use_slim_framework:
        validation_data_dir = data_dir
    else:
        validation_data_dir = os.path.join(data_dir, 'validation')
    prepare_data_in_tfrecord_format(val, validation_data_dir, split_name='validation',
                             output_file_prefix=output_file_prefix,
                             ratio_examples_each_class=1, duplicate=duplicate_training_data, verbose=verbose,
                             num_tf_file=num_tf_file,
                             num_threads=num_threads, version=version)
    print("Done")


def parse_image_record(image_record):
    features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(image_record, features=features)

    label = tf.cast(parsed_features['image/class/label'], tf.int32)
    image_height = tf.cast(parsed_features['image/height'], tf.int32)
    image_width = tf.cast(parsed_features['image/width'], tf.int32)

    image_raw = tf.image.decode_image(parsed_features['image/encoded'], channels=3)
    image_raw = tf.reshape(image_raw, [image_height, image_width, 3])
    image_raw = tf.cast(image_raw, tf.float32)

    return image_raw, label, [image_height, image_width]
