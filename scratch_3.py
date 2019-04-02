import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.layers.core import Dense


def binarization(W):
    Wb = tf.where(tf.greater_equal(W, 0.), tf.ones_like(W), -tf.ones_like(W))
    return Wb


class Dense_bin(Dense):

  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(Dense_bin, self).__init__(units=units,
                                activation=activation,
                                use_bias=use_bias,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                activity_regularizer=activity_regularizer,
                                kernel_constraint=kernel_constraint,
                                bias_constraint=bias_constraint,
                                trainable=trainable,
                                name=name,
                                **kwargs)

  def build(self, input_shape):
      super(Dense_bin, self).build(input_shape)
      self.kernel_t = tf.get_variable('kernel_t',
                                      shape=self.kernel.get_shape(),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=False,
                                      dtype=self.dtype)
      self.kernel.grad_update_var = self.kernel_t

  def call(self, inputs):
      assign_t = tf.assign(self.kernel.grad_update_var, self.kernel)
      with tf.control_dependencies([assign_t]):
          assign_b = tf.assign(self.kernel, binarization(self.kernel))
      with tf.control_dependencies([assign_t, assign_b]):
          return super(Dense_bin, self).call(inputs)
      '''
      assign_t = tf.assign(self.kernel_t, tf.clip_by_value(self.kernel_t, -1, 1))
      with tf.control_dependencies([assign_t]):
          assign_b = tf.assign(self.kernel, binarization(self.kernel_t))
      with tf.control_dependencies([assign_t, assign_b]):
          return super(Dense_bin, self).call(inputs)
      '''

def dense_bin(
    inputs, units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None):

    layer = Dense_bin(units,
                  activation=activation,
                  use_bias=use_bias,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer,
                  activity_regularizer=activity_regularizer,
                  kernel_constraint=kernel_constraint,
                  bias_constraint=bias_constraint,
                  trainable=trainable,
                  name=name,
                  _scope=name,
                  _reuse=reuse)
    return layer.apply(inputs)


x = tf.placeholder(tf.float32, [None, 5], name='input')
y = tf.placeholder(tf.float32, [None, 2], name='output')
pred = dense_bin(inputs=x, units=2)

loss = tf.losses.mean_squared_error(y, pred)

l_r = 1e-3
optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_r)
grads = optimizer.compute_gradients(loss, tf.trainable_variables())
grads_for_update = [(g, v.grad_update_var)
                    if hasattr(v, 'grad_update_var') else (g, v) for g, v in grads]
train = optimizer.apply_gradients(grads_for_update)

inp = np.random.uniform(0, 1, size=(5, 5))
out = np.random.randint(10, size=(5, 2))
vars = tf.trainable_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        vars_vals = sess.run(vars)
        print(vars_vals[0])
        print('-'*20)

        gradients = sess.run(grads, feed_dict={x: inp, y: out})

        gradients_for_update = sess.run(grads_for_update, feed_dict={x: inp, y: out})
        print(gradients_for_update[0][1] - l_r * gradients_for_update[0][0])
        print('-' * 20)

        sess.run(train, feed_dict={x: inp, y: out})
        vars_vals = sess.run(vars)
        print(vars_vals[0])
        print('-' * 20)
