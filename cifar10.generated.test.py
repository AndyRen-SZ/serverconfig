import os

import tensorflow as tf

import cifar10_generateds  as generate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
img, label = generate.read_test()
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print("1")
    images, labels = sess.run([img, label])
    print(images.shape)
    print(type(images))
    print(labels)

    generate.show_image(images[0])
    print(labels[0])
    # print(images)
    # print("2")
    # print(tf.shape(images))
    # print("3")
    # print(labels)
    # print(tf.shape(labels))
    coord.request_stop()
    coord.join(threads)
