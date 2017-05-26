import tensorflow as tf
import cv2
import numpy as np

# make background white or black (I forgot)
def gray_white(gray, thresh=50):
    binary_mask = np.zeros_like(gray)
    binary_mask[gray > thresh] = 1
    #print(gray.max(), gray.min())
    #plt.imshow(binary_mask)
    my_gray = gray.copy()
    my_gray[binary_mask == 1] = 255
    return my_gray

# processing pipeline
def process_image(img):
    img = cv2.resize(img, (28,28))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_white(gray, 40)
    gray = cv2.bitwise_not(gray)
    return gray

# load images
test_1 = cv2.imread('test_images/test_1.jpg')
test_2 = cv2.imread('test_images/test_2.jpg')
test_3 = cv2.imread('test_images/test_3.jpg')
test_4 = cv2.imread('test_images/test_4.jpg')
test_5 = cv2.imread('test_images/test_5.jpg')
processed_1 = process_image(test_1)
processed_2 = process_image(test_2)
processed_3 = process_image(test_3)
processed_4 = process_image(test_4)
processed_5 = process_image(test_5)
array_1 = processed_1.reshape(-1)
array_2 = processed_2.reshape(-1)
array_3 = processed_3.reshape(-1)
array_4 = processed_4.reshape(-1)
array_5 = processed_5.reshape(-1)
my_array = np.vstack((array_1, array_2, array_3, array_4, array_5))

# load the model
loaded_graph = tf.Graph()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./cnn_data.meta')
    new_saver.restore(sess, './cnn_data')
    predict_op = tf.get_collection('predict_op')[0]
    hparams = tf.get_collection("hparams")
    x = hparams[0]
    keep_prob = hparams[1]
    predicted_logits = sess.run(predict_op,feed_dict = {x: my_array, keep_prob: 1.0})

print(np.argmax(predicted_logits, axis=1))