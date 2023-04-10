import os

import numpy as np
import tensorflow as tf


def filter_item(data):
    features = tf.io.parse_single_example(data, {
        'label': tf.io.FixedLenFeature([], tf.string),
        'img_id': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    })
    image = tf.io.decode_raw(features['image_raw'], out_type=tf.uint8)
    return image.shape[0] == 150528


def decode_image(record_bytes):
    features = tf.io.parse_single_example(record_bytes, {
        'label': tf.io.FixedLenFeature([], tf.string),
        'img_id': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    })

    image = tf.io.decode_raw(features['image_raw'], out_type=tf.uint8)

    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')

    target = tf.cast(features['img_id'], tf.int64)
    print(target)

    return image, target


def get_dataset(tfrecords, bs=64, val=False, debug=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # Initialize dataset with TFRecords
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                      compression_type='GZIP')

    # Decode mapping
    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not val:
        dataset = dataset.with_options(ignore_order)
        if not debug:
            dataset = dataset.shuffle(100000)
        dataset = dataset.repeat()

    dataset = dataset.batch(bs, drop_remainder=not val)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
