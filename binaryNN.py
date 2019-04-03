from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import b_layer as binary_layer

# load dataset and preprocess
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

for i in range(mnist.train.images.shape[0]):
    mnist.train.images[i] = mnist.train.images[i] * 2 - 1
for i in range(mnist.validation.images.shape[0]):
    mnist.validation.images[i] = mnist.validation.images[i] * 2 - 1
'''
for i in range(mnist.train.labels.shape[0]):
    mnist.train.labels[i] = mnist.train.labels[i] * 2 - 1
for i in range(mnist.validation.labels.shape[0]):
    mnist.validation.labels[i] = mnist.validation.labels[i] * 2 - 1
'''

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

loss = tf.losses.softmax_cross_entropy(outputs, pred)
tf.summary.scalar('loss', loss)

l_r = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate=l_r)
grads = optimizer.compute_gradients(loss, tf.trainable_variables())
train = optimizer.apply_gradients(grads)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
val_writer = tf.summary.FileWriter('./logdir/val', accuracy.graph)
saver = tf.train.Saver()

# set hyper-parameters
batch_size = 100
epochs = 500
old_acc = 0

# training and validation
with tf.Session() as sess:
    print('*****************Training Start!*****************')
    sess.run(tf.global_variables_initializer())
    for m in range(epochs):
        iterations = int(mnist.train.num_examples/batch_size)
        for i in range(iterations):

            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = np.reshape(batch_x, [batch_size, 28, 28, 1])
            _, loss_train, summary = sess.run([train, loss, merged], {inputs: batch_x, outputs: batch_y})
            train_writer.add_summary(summary, i+m*iterations)

        val_x = np.reshape(mnist.validation.images, [5000, 28, 28, 1])
        val_y = mnist.validation.labels
        val_acc, summary = sess.run([accuracy, merged], {inputs: val_x, outputs: val_y})
        val_writer.add_summary(summary, m)
        print('Epoch: {}'.format(m + 1),
              'Train_loss: {:.3f}'.format(loss_train),
              'Val_accuracy: {:.3f}'.format(val_acc))

        if val_acc > old_acc:
            old_acc = val_acc
            saver.save(sess, './model/final.ckpt')

print('*****************Training End!*****************')
train_writer.close()
val_writer.close()
