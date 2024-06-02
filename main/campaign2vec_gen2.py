import os
import glob
import logging
import numpy as np
import pandas as pd
from tensorflow.python.framework.ops import disable_eager_execution
from datetime import datetime
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset


def generate_dataset(train_files_list, valid_files_list, test_files_list, batch_size=1024):
    """
    Return tensorflow Dataset object for training
    :param List[String] train_files_list: list of parquet file paths to use for training
    :param List[String] valid_files_list: list of parquet file paths to use for validation
    :param List[String] test_files_list: list of parquet file paths to use for test
    """
    BUFFER_SIZE = 1000
    dataset = load_dataset("parquet", data_files={'train': train_files_list, 'valid': valid_files_list,
                                                  'test': test_files_list}, streaming=True)
    dataset = dataset.select_columns(["v_context", "v_target", "label"])

    def gen(split):
        for item in dataset[split]:
            yield {'v_context': item['v_context']['values'], 'v_target': item['v_target']['values']}, item['label']

    tf_train_dataset = tf.data.Dataset.from_generator(lambda: gen('train'),
                                                      output_signature=({
                                                                            'v_context': tf.TensorSpec(shape=(1152,),
                                                                                                       dtype=tf.float16),
                                                                            'v_target': tf.TensorSpec(shape=(384,),
                                                                                                      dtype=tf.float16)},
                                                                        tf.TensorSpec(shape=(), dtype=tf.int32))) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    tf_valid_dataset = tf.data.Dataset.from_generator(lambda: gen('valid'),
                                                      output_signature=({
                                                                            'v_context': tf.TensorSpec(shape=(1152,),
                                                                                                       dtype=tf.float16),
                                                                            'v_target': tf.TensorSpec(shape=(384,),
                                                                                                      dtype=tf.float16)},
                                                                        tf.TensorSpec(shape=(), dtype=tf.int32))) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    tf_test_dataset = tf.data.Dataset.from_generator(lambda: gen('test'),
                                                     output_signature=({
                                                                           'v_context': tf.TensorSpec(shape=(1152,),
                                                                                                      dtype=tf.float16),
                                                                           'v_target': tf.TensorSpec(shape=(384,),
                                                                                                     dtype=tf.float16)},
                                                                       tf.TensorSpec(shape=(), dtype=tf.int32))) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return tf_train_dataset, tf_valid_dataset, tf_test_dataset


def create_model():
    v_context = tf.keras.Input(shape=(1152,), name='v_context', dtype='float16')
    v_target = tf.keras.Input(shape=(384,), name='v_target', dtype='float16')
    repeated = tf.keras.layers.RepeatVector(3)(v_target)
    repeated = tf.keras.layers.Reshape((1152,))(repeated)
    merged = tf.keras.layers.Concatenate()([v_context, repeated])
    # merged = tf.keras.layers.Dot(name='dot', normalize=True, axes=1)([x1, repeated])
    merged = tf.keras.layers.Dense(100, activation='relu', name='merged')(merged)
    y = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(merged)
    model = tf.keras.Model(inputs=[v_context, v_target], outputs=y)
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'],
    )
    return model


def get_callbacks(checkpoint_dir='', valid_data=None, save_every_num_batch=100000):

    class InterEpochValidation(tf.keras.callbacks.Callback):

        def __init__(self, valid_data, save_every_num_batch):
            super().__init__()
            self.validation_data = valid_data
            self.save_every_num_batch = save_every_num_batch

        def on_train_batch_end(self, batch, logs=None):
            if (batch + 1) % self.save_every_num_batch == 0:
                m = tf.keras.metrics.BinaryAccuracy()
                try:
                    for idx, example in enumerate(self.validation_data):
                        m.update_state(example[1], self.model.predict(example[0]))
                        if idx >= 100:
                            break
                    print(f"Validation accuracy: {m.result().numpy().mean()}")
                except Exception as e:
                    print(e)


    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True,
                                                             monitor='val_loss', mode='min',
                                                             save_freq=save_every_num_batch)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=1,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0
    )

    return [early_stopping, InterEpochValidation(valid_data, save_every_num_batch), checkpoint_callback]


model = create_model()

tf_train_dataset, tf_valid_dataset, tf_test_dataset = \
    generate_dataset(train_files_list=['model_data/training_set_parquet/*.parquet'],
                     valid_files_list=['model_data/validation_set_parquet/*.parquet'],
                     test_files_list=['model_data/test_set_parquet/*.parquet'],
                     batch_size=32)

callbacks = get_callbacks('model_data/checkpoints', tf_valid_dataset, 20_000)
model.fit(tf_train_dataset,
          steps_per_epoch=(1292462 // 32) + 1,
          validation_data=tf_valid_dataset,
          epochs=10,
          callbacks=callbacks
          )

model.evaluate(tf_test_dataset)
model.save("model_data/models/mode_v0.h5")
print('done')

# model = tf.keras.saving.load_model('model_data/models/mode_v0_1.h5')
# model.evaluate(tf_test_dataset)