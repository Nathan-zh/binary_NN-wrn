from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
for i in range(mnist.test.images.shape[0]):
    mnist.test.images[i] = mnist.test.images[i] * 2 - 1
'''
tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
outputs = tf.placeholder(tf.float32, [None, 10], name='output')

# Layer 1: conv
x = tf.layers.conv2d(inputs=inputs,
                     filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same')
x = tf.nn.relu(x)
x = tf.layers.batch_normalization(x)

# Layer 2: conv
x = tf.layers.conv2d(inputs=inputs,
                     filters=64,
                     kernel_size=(7, 7),
                     strides=(3, 3))
x = tf.nn.relu(x)
x = tf.layers.batch_normalization(x)
x = tf.layers.dropout(x, 0.4)

x = tf.transpose(x, perm=[0, 3, 1, 2])
x = tf.layers.flatten(x)

# Layer 3: FC
x = tf.layers.dense(x, units=2048)
x = tf.nn.relu(x)
x = tf.layers.batch_normalization(x)
x = tf.layers.dropout(x, 0.5)

# Layer 4: FC
x = tf.layers.dense(x, units=512)
x = tf.nn.relu(x)
x = tf.layers.batch_normalization(x)
x = tf.layers.dropout(x, 0.4)

# Layer 5: FC
x = tf.layers.dense(x, units=128)
x = tf.nn.relu(x)
x = tf.layers.batch_normalization(x)
x = tf.layers.dropout(x, 0.3)

pred = tf.layers.dense(x, units=10)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
vars = tf.trainable_variables()

print('*****************Test stage*****************')
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './model/final1.ckpt')
    vars_vals = sess.run(vars)
    '''
    # save model parameters to files
    for i in range(len(vars)):
        if i == 0:
            para = vars_vals[i].transpose((3, 2, 0, 1)).reshape([4, 49])
        else:
            para = vars_vals[i]
        filename = vars[i].name
        filename = filename.replace('/', ':')
        np.savetxt('./data/{}.txt'.format(filename), para)
    '''
    test_x = np.reshape(mnist.test.images, [10000, 28, 28, 1])
    test_y = mnist.test.labels
    prediction, test_acc = sess.run([pred, accuracy], {inputs: test_x, outputs: test_y})

print('Test accuracy: {:.4f}'.format(test_acc))
