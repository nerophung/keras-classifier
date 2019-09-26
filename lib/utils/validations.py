from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import glob


def validate_presence(object, message=""):
    if not object:
        raise ValueError(message)

    return True


def validate_existence_path(path, message=""):
    if not os.path.exists(path):
        raise ValueError(message)

    return True


def validate_valid_path(path, message=""):
    path_list = glob.glob(path)
    if len(path_list) == 0:
        raise ValueError(message)

    return path_list


def validate_presence_and_make_directory(dir_path, message=""):
    validate_presence(dir_path, message=message)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


if __name__ == '__main__':
    pass