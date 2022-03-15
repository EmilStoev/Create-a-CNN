from __future__ import division, print_function, unicode_literals #compatibility
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Clear tensorflow's and reset seed
def reset_graph(seed=None):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
digits = np.concatenate((X_train, X_test))
labels = np.concatenate((y_train, y_test))

# Pre-processing the data
t_digits = digits.astype(np.float32).reshape(-1, 28*28) / 255.0
t_labels = labels.astype(np.int32)

# MNIST's specification
height = 28
width = 28
channels = 1

reset_graph()

# Create TensorFlow's placeholders for digits and labels
X = tf.placeholder(tf.float32, shape=[None, height * width], name="X")
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
y = tf.placeholder(tf.int32, shape=[None], name="y")

# Construct 2D convolutional layers
conv1 = tf.layers.conv2d(X_reshaped, filters=20, kernel_size=3, strides=1,
                         padding="SAME", activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=40, kernel_size=3, strides=2,
                         padding="SAME", activation=tf.nn.relu, name="conv2")

# Create a max pooling layer
pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
pool3_flat = tf.reshape(pool3, shape=[-1, 40 * 7 * 7])

# Followed by layer of fully-connected neurons
fc1 = tf.layers.dense(pool3_flat, 50, activation=tf.nn.relu, name="fc1")
logits = tf.layers.dense(fc1, 10, name="output")

# Use mean softmax cross entropy as a loss function
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(xentropy)

# Use Adam Optimiser to train CNN
training_op = tf.train.AdamOptimizer().minimize(loss,
                                                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

# Define accuracy measure
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# Train the CNN batch by batch
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(10):
        for X_batch, y_batch in shuffle_batch(t_digits, t_labels, 50):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: t_digits, y: t_labels})
        print(epoch, "Accuracy:", acc_batch)

    # save the trained model
    save_path = tf.train.Saver().save(sess, "./trained_mnist_cnn.ckpt")

# random one digit for test CNN's prediction
rnd_id = np.random.randint(0, len(digits))

# visualise the digit
plt.figure()
plt.imshow(digits[rnd_id])
plt.colorbar()
plt.grid(False)

# load the trained model and use to predict
with tf.Session() as sess:
    tf.train.Saver().restore(sess, "./trained_mnist_cnn.ckpt")
    Z = logits.eval(feed_dict={X: t_digits[rnd_id].reshape(1, 28*28)})
    y_pred = np.argmax(Z, axis=1)

print("Predicted class: ", y_pred)
print("Actual class: ", labels[rnd_id])

iterations = 100 # repeat for 100 iterations to check accuracy
accuracy = 0
for i in range(0, iterations):
    rnd_id = np.random.randint(0, len(digits))
    # Visualisation is removed as it requires too much memory. If needed the below code can be uncommented and checked
    # plt.figure()
    # plt.imshow(digits[rnd_id])
    # plt.colorbar()
    # plt.grid(False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, "./trained_mnist_cnn.ckpt")
        Z = logits.eval(feed_dict={X: t_digits[rnd_id].reshape(1, 28 * 28)})
        y_pred = np.argmax(Z, axis=1)

    print("Predicted class: ", y_pred)
    print("Actual class: ", labels[rnd_id])
    if y_pred == labels[rnd_id]:
        accuracy += 1

total = (accuracy / iterations) * 100
print('Total accuracy of model', total, '%')