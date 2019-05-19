# coding: utf-8
# generate cifar10 数据集,and read it
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# test train file pattern
TEST_FILE_PATTERN = "D:/BaiduNetdiskDownload/ai/cifar-10/test/*/batch_*_num_*.jpg"
TRAIN_FILE_PATTERN = "D:/BaiduNetdiskDownload/ai/cifar-10/train/*/batch_*_num_*.jpg"

# test train dataSet pattern
TEST_DATA_SET_PATTERN = "C:/Users/Administrator/PycharmProjects/mytensorflow/data.cifar10-test-*-of-*"
TRAIN_DATA_SET_PATTERN = "C:/Users/Administrator/PycharmProjects/mytensorflow/data.cifar10-train-*-of-*"

# image size
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNEL = 3


def show_image(img_to_show):
    plt.figure(1)
    plt.imshow(img_to_show)
    plt.show()


def write_record(writer, label, img_raw):
    labels = jude_label(label)
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw.tobytes()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
    }))
    writer.write(example.SerializeToString())


def jude_label(key_byte):
    labels = [0] * 10
    key_ = str(key_byte)
    if 'airplane' in key_:
        labels[0] = 1
    elif 'automobile' in key_:
        labels[1] = 1
    elif 'bird' in key_:
        labels[2] = 1
    elif 'cat' in key_:
        labels[3] = 1
    elif 'deer' in key_:
        labels[4] = 1
    elif 'dog' in key_:
        labels[5] = 1
    elif 'frog' in key_:
        labels[6] = 1
    elif 'horse' in key_:
        labels[7] = 1
    elif 'ship' in key_:
        labels[8] = 1
    elif 'truck' in key_:
        labels[9] = 1
    return labels


def generate_record(file_pattern, save_file_name, total_num_shards, instances_per_shard=10000):
    # 所有图片的列表
    file_list = tf.train.match_filenames_once(file_pattern)

    # 创建输入队列,默认顺序打乱
    file_queue = tf.train.string_input_producer(file_list, shuffle=False, num_epochs=1)
    key, image = tf.WholeFileReader().read(file_queue)

    # 解码成tf中图像格式
    image = tf.image.decode_jpeg(image)
    image_float32_op = tf.image.convert_image_dtype(image, dtype=tf.float32)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(total_num_shards):
            filename = (save_file_name + '-%.5d-of-%.5d' % (i, total_num_shards))
            writer = tf.python_io.TFRecordWriter(filename)
            for j in range(instances_per_shard):
                print(j)
                key_label, imgFloat32 = sess.run([key, image_float32_op])
                write_record(writer, key_label, np.reshape(imgFloat32, (1, -1)))
            writer.close()
        coord.request_stop()
        coord.join(threads)


def read_test():
    reader = tf.TFRecordReader()
    # 以下代码不能运行 why
    # test_data_set_file_list = tf.train.match_filenames_once(TEST_DATA_SET_PATTERN)
    # filename_queue = tf.train.string_input_producer(test_data_set_file_list, shuffle=False, num_epochs=1)
    filename_queue = tf.train.string_input_producer(
        ["C:/Users/Administrator/PycharmProjects/mytensorflow/data.cifar10-test-00000-of-00001"], shuffle=False,
        num_epochs=1)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([10], tf.int64)
                                       })
    img = features['img_raw']
    img = tf.decode_raw(img, tf.float32)
    img = tf.reshape(img, [IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
    label = tf.cast(features['label'], tf.float32)
    min_after_dequeue = 10000
    batch_size = 10000
    capacity = min_after_dequeue + 3 * batch_size

    image_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                      batch_size=batch_size,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


def batch_read_train(batch_size):
    reader = tf.TFRecordReader()
    # 以下代码不能运行 why
    # test_data_set_file_list = tf.train.match_filenames_once(TEST_DATA_SET_PATTERN)
    # filename_queue = tf.train.string_input_producer(test_data_set_file_list, shuffle=False, num_epochs=1)
    filename_queue = tf.train.string_input_producer(
        ["C:/Users/Administrator/PycharmProjects/mytensorflow/data.cifar10-train-00000-of-00005",
         "C:/Users/Administrator/PycharmProjects/mytensorflow/data.cifar10-train-00001-of-00005",
         "C:/Users/Administrator/PycharmProjects/mytensorflow/data.cifar10-train-00002-of-00005",
         "C:/Users/Administrator/PycharmProjects/mytensorflow/data.cifar10-train-00003-of-00005",
         "C:/Users/Administrator/PycharmProjects/mytensorflow/data.cifar10-train-00004-of-00005"], shuffle=True,
        num_epochs=2)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([10], tf.int64)
                                       })
    img = features['img_raw']
    img = tf.decode_raw(img, tf.float32)
    img = tf.reshape(img, [IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
    label = tf.cast(features['label'], tf.float32)
    min_after_dequeue = 300
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                      batch_size=batch_size,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


def main():
    # generate_record(TEST_FILE_PATTERN, 'data.cifar10-test', 1, 10000)
    generate_record(TRAIN_FILE_PATTERN, 'data.cifar10-train', 5)


if __name__ == '__main__':
    main()
