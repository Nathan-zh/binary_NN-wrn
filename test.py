from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# load dataset and preprocess
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
for i in range(mnist.test.images.shape[0]):
    mnist.test.images[i] = mnist.test.images[i] * 2 - 1
'''

# tensor graph
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
outputs = tf.placeholder(tf.float32, [None, 10], name='output')

# Layer 1: conv
x = tf.layers.conv2d(inputs=inputs,
                     filters=32,
                     kernel_size=(7, 7),
                     strides=(3, 3))
x = tf.layers.batch_normalization(x)
x = tf.square(x)
x = tf.layers.dropout(x, 0.5)

# Layer 2: conv
x = tf.layers.conv2d(inputs=x,
                     filters=64,
                     kernel_size=(3, 3),
                     strides=(1, 1))
x = tf.layers.batch_normalization(x)
x = tf.layers.dropout(x, 0.4)

x = tf.transpose(x, perm=[0, 3, 1, 2])
x = tf.layers.flatten(x)

# Layer 3: FC
x = tf.layers.dense(x, units=2048)
x = tf.layers.batch_normalization(x)
x = tf.square(x)
x = tf.layers.dropout(x, 0.5)

# Layer 4: FC
x = tf.layers.dense(x, units=512)
x = tf.layers.batch_normalization(x)
x = tf.layers.dropout(x, 0.4)

# Layer 5: FC
x = tf.layers.dense(x, units=128)
x = tf.layers.batch_normalization(x)
x = tf.layers.dropout(x, 0.3)

pred = tf.layers.dense(x, units=10)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('*****************Test stage*****************')
vars_vals = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # read weights and bias
    vars_vals.append(np.loadtxt('./weights/conv2d_bin:kernel:0.txt').reshape([32, 1, 7, 7]).transpose((2, 3, 1, 0)))
    vars_vals.append(np.loadtxt('./weights/conv2d_bin:bias:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization:gamma:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization:beta:0.txt'))

    vars_vals.append(np.loadtxt('./weights/conv2d_bin_1:kernel:0.txt').reshape([64, 32, 3, 3]).transpose((2, 3, 1, 0)))
    vars_vals.append(np.loadtxt('./weights/conv2d_bin_1:bias:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization_bin:gamma:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization_bin:beta:0.txt'))

    vars_vals.append(np.loadtxt('./weights/dense_bin:kernel:0.txt'))
    vars_vals.append(np.loadtxt('./weights/dense_bin:bias:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization_1:gamma:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization_1:beta:0.txt'))

    vars_vals.append(np.loadtxt('./weights/dense_bin_1:kernel:0.txt'))
    vars_vals.append(np.loadtxt('./weights/dense_bin_1:bias:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization_bin_1:gamma:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization_bin_1:beta:0.txt'))

    vars_vals.append(np.loadtxt('./weights/dense_bin_2:kernel:0.txt'))
    vars_vals.append(np.loadtxt('./weights/dense_bin_2:bias:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization_bin_2:gamma:0.txt'))
    vars_vals.append(np.loadtxt('./weights/batch_normalization_bin_2:beta:0.txt'))

    vars_vals.append(np.loadtxt('./weights/dense_bin_3:kernel:0.txt'))
    vars_vals.append(np.loadtxt('./weights/dense_bin_3:bias:0.txt'))

    # Binarize weights and assignment
    for i in range(len(vars_vals)):
        if i % 2 == 0 and i != 2 and i != 10:
            vars_vals[i][vars_vals[i] >= 0.] = 1.
            vars_vals[i][vars_vals[i] < 0.] = -1.
        sess.run(tf.assign(tf.trainable_variables()[i], vars_vals[i]))
        #assert sess.run(tf.trainable_variables()[i]).all() == vars_vals[i].all()

    test_x = np.reshape(mnist.test.images, [10000, 28, 28, 1])
    test_y = mnist.test.labels
    test_acc = sess.run(accuracy, {inputs: test_x, outputs: test_y})

print('Test accuracy: {:.4f}'.format(test_acc))
