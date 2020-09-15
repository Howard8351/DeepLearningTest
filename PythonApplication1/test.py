import tensorflow as tf

a = tf.placeholder(tf.float32, [None, 10, 10, 3], name="fake_pool_A")
b = 0