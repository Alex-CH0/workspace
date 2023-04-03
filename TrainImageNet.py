#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import tensorflow as tf
from ImageNet21kSemanticSoftmax import ImageNet21kSemanticSoftmax
from NextVit import NextViT
from SemanticSoftmaxLoss import SemanticSoftmaxLoss
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NORM_EPS = 1e-5
SEED = 42
strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE = 64 * strategy.num_replicas_in_sync

cwd = os.getcwd()
tfrecord_train = os.path.join(cwd, 'train_tfrec.tfrecords')
tfrecord_val = os.path.join(cwd, 'val_tfrec.tfrecords')

TRAIN_SIZE = 11060223
VAL_SIZE = 522500

TRAIN_STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VAL_STEPS_PER_EPOCH = VAL_SIZE // BATCH_SIZE
print(f'TRAIN_STEPS_PER_EPOCH: {TRAIN_STEPS_PER_EPOCH}, VAL_STEPS_PER_EPOCH: {VAL_STEPS_PER_EPOCH}')

with strategy.scope():
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, num_classes=10450)


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


def get_dataset(tfrecords, bs=BATCH_SIZE, val=False, debug=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # Initialize dataset with TFRecords
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                      compression_type='GZIP')

    # dataset = dataset.filter(filter_item)

    # Decode mapping
    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not val:
        dataset = dataset.with_options(ignore_order)
        if not debug:
            dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()

    dataset = dataset.batch(bs, drop_remainder=not val)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


train_dataset = get_dataset(tfrecord_train, val=False, debug=False)
val_dataset = get_dataset(tfrecord_val, val=True, debug=False)

# Sanity check, image and label statistics
X_batch, y_batch = next(iter(train_dataset))
image = X_batch.numpy()
print(f'image shape: {image.shape}, y_batch shape: {y_batch.shape}')
print(f'image dtype: {image.dtype}, y_batch dtype: {y_batch.dtype}')
print(f'image min: {image.min():.2f}, max: {image.max():.2f}')

semantic_softmax_processor = ImageNet21kSemanticSoftmax(label_smooth=0.2)
with strategy.scope():
    loss = SemanticSoftmaxLoss(semantic_softmax_processor)

training_callbacks = [
    tf.keras.callbacks.CSVLogger('training_history.csv'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5',
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=7,
                                     mode='min')
]

model.compile(optimizer='adam', loss=loss,
              metrics=['acc'])
model.fit(train_dataset
          , steps_per_epoch=TRAIN_STEPS_PER_EPOCH
          , validation_data=val_dataset
          , epochs=100)
