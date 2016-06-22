from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

_module = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'my_pad.so'))
m_pad = _module.m_pad

