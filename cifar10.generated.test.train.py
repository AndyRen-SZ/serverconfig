import os

import tensorflow as tf

import cifar10_generateds  as generate

BATCH_SIZE_NUMBER = 100

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
img, label = generate.read_test()
timg, tlabel = generate.batch_read_train(BATCH_SIZE_NUMBER)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # print("1")
    # images, labels = sess.run([img, label])
    # print(images.shape)
    # print(type(images))
    # print(labels)
    #
    # generate.show_image(images[0])
    # print(labels[0])
    train_images, train_labels = sess.run([timg, tlabel])
    print(type(train_images))
    print(train_images.shape)
    generate.show_image(train_images[0])
    print(train_labels[0])

    coord.request_stop()
    coord.join(threads)
