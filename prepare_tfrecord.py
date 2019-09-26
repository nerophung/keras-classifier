from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import lib.utils.validations as validations
from lib.utils.tfrecord import convert_dataset_to_tfrecords

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', '', 'Path to the dataset directory')
tf.app.flags.DEFINE_string('data_dir', '', 'Path to the data (in tfrecord format) directory')
tf.app.flags.DEFINE_string('output_file_prefix', '', 'prefix for output tfrecord data file name')
tf.app.flags.DEFINE_float('ratio_train_data', 0.8, 'The proportion of data that used for training')
tf.app.flags.DEFINE_bool('duplicate_training_data', False,
                         'Whether to duplicate data when the number of examples in classes are not equal')
tf.app.flags.DEFINE_bool('use_slim_framework', True, 'If true, prepare data for using with Slim framework')
tf.app.flags.DEFINE_bool('verbose', False, '')
tf.app.flags.DEFINE_integer('num_tf_file', 10, 'Number of tf record files')
tf.app.flags.DEFINE_integer('num_threads', 3, 'Number of threads')


if __name__ == '__main__':
    validations.validate_presence(FLAGS.dataset_dir, message="Please enter the path to the dataset directory")
    validations.validate_existence_path(FLAGS.dataset_dir, message=("Directory '%s' is not exist" % FLAGS.dataset_dir))
    validations.validate_presence_and_make_directory(FLAGS.data_dir,
                                                     message="Please enter the path to the data directory")

    if not os.listdir(FLAGS.data_dir):
        convert_dataset_to_tfrecords(FLAGS.dataset_dir, FLAGS.data_dir, ratio_train_data=FLAGS.ratio_train_data,
                                     output_file_prefix=FLAGS.output_file_prefix,
                                     duplicate_training_data=FLAGS.duplicate_training_data,
                                     use_slim_framework=FLAGS.use_slim_framework,
                                     verbose=FLAGS.verbose,
                                     num_tf_file=FLAGS.num_tf_file,
                                     num_threads=FLAGS.num_threads)