# import libraries
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# create a session
sess = tf.InteractiveSession()
# load the data
mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)

def weight_variable(shape):
    # initialize weights, normalize it to make it easier to converge
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # not 0 in case of never being activated
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def cnn(x):
    # return the logits
    
    # [None, 784] ==> [None, 28, 28, 1]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # convolutional layer 1
    conv1_w = weight_variable([5, 5, 1, 32])
    conv1_b = bias_variable([32])

    conv1 = tf.nn.relu(conv2d(x_image, conv1_w) + conv1_b)
    conv1 = max_pool_2x2(conv1)

    # convolutional layer 2
    conv2_w = weight_variable([5, 5, 32, 64])
    conv2_b = bias_variable([64])

    conv2 = tf.nn.relu(conv2d(conv1, conv2_w) + conv2_b)
    conv2 = max_pool_2x2(conv2)

    # flatten layer
    fcl_w = weight_variable([7 * 7 * 64, 1024])
    fcl_b = bias_variable([1024])

    flatten = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fcl = tf.nn.relu(tf.matmul(flatten, fcl_w) + fcl_b)

    # dropout layer
    fc1 = tf.nn.dropout(fcl, keep_prob)

    # output layer
    fc2_w = weight_variable([1024, 10])
    fc2_b = bias_variable([10])

    logits = tf.nn.softmax(tf.matmul(fc1, fc2_w) + fc2_b)
    
    return logits

# create placeholders
x = tf.placeholder(tf.float32, shape = [None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape = [None, 10], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
logits = cnn(x)

# training and evaluation operations
cross_entropy = -tf.reduce_sum(y_*tf.log(logits))
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# train the CNN
sess.run(tf.global_variables_initializer())
for i in range(2000):
    batch = mnist.train.next_batch(64)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("iter: %d, training accuracy: %g" % (i, train_accuracy))
    train_op.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.7})

print("test accuracy: %g" % accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# save the model
saver.save(sess, './cnn_data')
print("Model saved")

# close the session and release the RAM
sess.close()
logits = cnn(x)