from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class MyPadOpTest(tf.test.TestCase):
  def testMyPad(self):
    my_pad_module = tf.load_op_library('my_pad.so')
    with self.test_session():
      result = my_pad_module.m_pad([5, 4, 3, 2, 1], [[3, 4]])
      self.assertAllEqual(result.eval(), [0, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0, 0])


if __name__ == '__main__':
  tf.test.main()
