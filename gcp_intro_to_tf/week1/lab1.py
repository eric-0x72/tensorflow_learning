import tensorflow as tf
import numpy as np

print(tf.__version__)


#
# a = tf.constant([5, 3, 8])
# b = tf.constant([3, -1, 2])
# c = tf.add(a, b)
# print(c)
#
# # run session
# with tf.Session() as sess:
#     result = sess.run(c)
#     print(result)
#
# # use feed dict
# a = tf.placeholder(dtype=tf.int32, shape=(None,))  # batchsize x scalar
# b = tf.placeholder(dtype=tf.int32, shape=(None,))
# c = tf.add(a, b)
# with tf.Session() as sess:
#     result = sess.run(c, feed_dict={
#         a: [3, 4, 5],
#         b: [-1, 2, 3]
#     })
#     print(result)
#

def compute_area(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2

    areasq = s * (s - a) * (s - b) * (s - c)
    area = tf.sqrt(areasq)
    return area


# with tf.Session() as sess:
#     # pass in two triangles
#     area = compute_area(tf.constant([
#         [5.0, 3.0, 7.1],
#         [2.3, 4.1, 4.8],
#         [1, 1, 1]
#     ]))
#     result = sess.run(area)
#     print(result)


with tf.Session() as sess:
    sides = tf.placeholder(tf.float32, shape=(None, 3))  # placeholder here
    area = compute_area(sides)
    result = sess.run(area, feed_dict={sides: [[5.0, 3.0, 7.1],
                                               [2.3, 4.1, 4.8]]})
    print(result)
