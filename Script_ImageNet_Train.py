
import os

import numpy as np
import tensorflow as tf
from ImageNet21kSemanticSoftmax import ImageNet21kSemanticSoftmax
from NextVit import NextViT
from SemanticSoftmaxLoss import SemanticSoftmaxLoss
import DataGenerator as DataGen
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

with strategy.scope():
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, num_classes=10450, activation='softmax')


train_dataset = DataGen.get_dataset(tfrecord_train, bs=BATCH_SIZE, val=False, debug=False)
val_dataset = DataGen.get_dataset(tfrecord_val, bs=BATCH_SIZE, val=True, debug=False)


semantic_softmax_processor = ImageNet21kSemanticSoftmax(label_smooth=0.2)
with strategy.scope():
    loss = SemanticSoftmaxLoss(semantic_softmax_processor)

training_callbacks = [
    tf.keras.callbacks.CSVLogger(os.path.join(cwd, 'training_history.csv')),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cwd,'model.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5'),
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
          , epochs=100
          ,callbacks=training_callbacks
          ,validation_steps= VAL_STEPS_PER_EPOCH
         )
