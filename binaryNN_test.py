from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import b_layer as binary_layer

# load dataset and preprocess
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

for i in range(mnist.test.images.shape[0]):
    mnist.test.images[i] = mnist.test.images[i] * 2 - 1

# tensor graph
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
outputs = tf.placeholder(tf.float32, [None, 10], name='output')

x = binary_layer.conv2d(inputs=inputs, filters=4, kernel_size=(7, 7), strides=(3, 3))
x = tf.nn.tanh(x)
x = tf.transpose(x, perm=[0, 3, 1, 2])
x = tf.layers.flatten(x)
x = binary_layer.dense(x, units=64)
x = tf.nn.tanh(x)
pred = binary_layer.dense(x, units=10)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('*****************Test stage*****************')
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './model/final.ckpt')
    '''
    vars_vals = sess.run(vars)
    # save model parameters to files
    for i in range(len(vars)):

        if i == 0:
            para = vars_vals[i].transpose((3, 2, 0, 1)).reshape([4, 49])
        else:
            para = vars_vals[i]
        filename = vars[i].name
        filename = filename.replace('/', ':')
        np.savetxt('./{}.txt'.format(filename), para)
    '''
    test_x = np.reshape(mnist.test.images, [10000, 28, 28, 1])
    test_y = mnist.test.labels
    test_acc = sess.run(accuracy, {inputs: test_x, outputs: test_y})

print('Test accuracy: {:.4f}'.format(test_acc))
