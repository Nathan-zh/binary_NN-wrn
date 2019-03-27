from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
#from Binary_Dense import Binary_Dense


def binarize(aa2):
    aa1 = []
    for i in range(len(aa2)):
        if i % 2 == 0:
            aa1.append(np.sign(aa2[i]))
    return aa1


def clip(bb1):
    bb1[bb1 > 1] = 1
    bb1[bb1 < -1] = -1


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
outputs = tf.placeholder(tf.float32, [None, 10], name='output')

x = tf.layers.conv2d(inputs=inputs,
                     filters=4,
                     kernel_size=(7, 7),
                     strides=(3, 3))
x = tf.square(x)
x = tf.transpose(x, perm=[0, 3, 1, 2])
x = tf.layers.flatten(x)
x = tf.layers.dense(x, units=64)
x = tf.square(x)
pred = tf.layers.dense(x, units=10)

loss = tf.losses.softmax_cross_entropy(outputs, pred)
tf.summary.scalar('loss', loss)

#vars = tf.trainable_variables()
train = tf.train.AdamOptimizer(learning_rate=1e-4)
grads = train.compute_gradients(loss, tf.trainable_variables())
train1 = train.apply_gradients(grads)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
val_writer = tf.summary.FileWriter('./logdir/val', accuracy.graph)
saver = tf.train.Saver()

batch_size = 1000
epochs = 500
with tf.Session() as sess:
    print('*****************Training Start!*****************')
    sess.run(tf.global_variables_initializer())
    for m in range(epochs):
        iterations = int(mnist.train.num_examples/batch_size)
        for i in range(iterations):
            vars_vals = sess.run(tf.trainable_variables())
            #print(sess.run(tf.trainable_variables()[0][:, :, 0, 0]))
            #print('**********************************************************')
            b_weights = binarize(vars_vals)
            sess.run(tf.assign(tf.trainable_variables()[0], b_weights[0]))
            sess.run(tf.assign(tf.trainable_variables()[2], b_weights[1]))
            sess.run(tf.assign(tf.trainable_variables()[4], b_weights[2]))
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = np.reshape(batch_x, [batch_size, 28, 28, 1])
            gradients, loss_train, prediction, summary = sess.run([grads, loss, pred, merged], {inputs: batch_x, outputs: batch_y})
            #print(- gradients[0][0][:, :, 0, 0] * 1e-4)
            #print('**********************************************************')
            '''
            for ii in range(len(tf.trainable_variables())):
                new_weights = - gradients[ii][0] * 1e-4 + vars_vals[ii]
                if ii % 2 == 0:
                    clip(new_weights)
                sess.run(tf.assign(tf.trainable_variables()[ii], new_weights))
            '''
            sess.run(tf.assign(grads[0][1], vars_vals[0]))
            sess.run(tf.assign(grads[2][1], vars_vals[2]))
            sess.run(tf.assign(grads[4][1], vars_vals[4]))

            sess.run(train1, {inputs: batch_x, outputs: batch_y})

            #print(sess.run(tf.trainable_variables()[0][:, :, 0, 0]))
            #print('**********************************************************')
            train_writer.add_summary(summary, i+m*iterations)

        #print('prediction is : ', prediction[0, :])
        vars_vals = sess.run(tf.trainable_variables())
        b_weights = binarize(vars_vals)
        sess.run(tf.assign(tf.trainable_variables()[0], b_weights[0]))
        sess.run(tf.assign(tf.trainable_variables()[2], b_weights[1]))
        sess.run(tf.assign(tf.trainable_variables()[4], b_weights[2]))
        val_x = np.reshape(mnist.validation.images, [5000, 28, 28, 1])
        val_y = mnist.validation.labels
        val_acc, summary = sess.run([accuracy, merged], {inputs: val_x, outputs: val_y})
        val_writer.add_summary(summary, m)
        print('Epoch: {}'.format(m + 1),
              'Train_loss: {:.3f}'.format(loss_train),
              'Val_accuracy: {:.3f}'.format(val_acc))
        sess.run(tf.assign(tf.trainable_variables()[0], vars_vals[0]))
        sess.run(tf.assign(tf.trainable_variables()[2], vars_vals[2]))
        sess.run(tf.assign(tf.trainable_variables()[4], vars_vals[4]))
        for iii in range(len(tf.trainable_variables())):
            assert vars_vals[iii].all() <= 1 and vars_vals[iii].all() >= -1

    saver.save(sess, './model/final.ckpt')

print('*****************Training End!*****************')
train_writer.close()
val_writer.close()
