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


def generate_sample():
    pd.DataFrame({'x1': np.random.random((100, 21)).tolist(),
                  'x2': np.random.random((100, 7)).tolist(),
                  'label': np.random.randint(0, 2, 100).tolist()}).to_parquet(
        'tests/data/word2vec/ver2_parquet/1.parquet')


def generate_dataset(train_files_list, valid_files_list, test_files_list, batch_size=2):
    """
    Return tensorflow Dataset object for training
    :param List[String] train_files_list: list of parquet file paths to use for training
    :param List[String] valid_files_list: list of parquet file paths to use for validation
    :param List[String] test_files_list: list of parquet file paths to use for test

    """

    dataset = load_dataset("parquet", data_files={'train': train_files_list, 'valid': valid_files_list,
                                                  'test': test_files_list}, streaming=True)

    def gen(split):
        for item in dataset[split]:
            yield {'x1': item['x1'], 'x2': item['x2']}, item['label']

    tf_train_dataset = tf.data.Dataset.from_generator(lambda: gen('train'),
                                                      output_signature=(
                                                          {'x1': tf.TensorSpec(shape=(21,), dtype=tf.float16),
                                                           'x2': tf.TensorSpec(shape=(7,), dtype=tf.float16)},
                                                          tf.TensorSpec(shape=(), dtype=tf.int32))) \
        .batch(batch_size)

    tf_valid_dataset = tf.data.Dataset.from_generator(lambda: gen('valid'),
                                                      output_signature=(
                                                          {'x1': tf.TensorSpec(shape=(21,), dtype=tf.float16),
                                                           'x2': tf.TensorSpec(shape=(7,), dtype=tf.float16)},
                                                          tf.TensorSpec(shape=(), dtype=tf.int32))) \
        .batch(batch_size)

    tf_test_dataset = tf.data.Dataset.from_generator(lambda: gen('test'),
                                                      output_signature=(
                                                          {'x1': tf.TensorSpec(shape=(21,), dtype=tf.float16),
                                                           'x2': tf.TensorSpec(shape=(7,), dtype=tf.float16)},
                                                          tf.TensorSpec(shape=(), dtype=tf.int32))) \
        .batch(batch_size)

    return tf_train_dataset, tf_valid_dataset, tf_test_dataset


def create_model():
    x1 = tf.keras.Input(shape=(21,), name='x1', dtype='float16')
    x2 = tf.keras.Input(shape=(7,), name='x2', dtype='float16')
    repeated = tf.keras.layers.RepeatVector(3)(x2)
    repeated = tf.keras.layers.Reshape((21,))(repeated)
    merged = tf.keras.layers.Concatenate()([x1, repeated])
    # merged = tf.keras.layers.Dot(name='dot', normalize=True, axes=1)([x1, repeated])
    merged = tf.keras.layers.Dense(100, activation='relu', name='merged')(merged)
    y = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(merged)
    model = tf.keras.Model(inputs=[x1, x2], outputs=y)
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'],
    )
    return model


def get_callbacks(checkpoint_dir='', valid_data=None):
    class InterEpochValidation(tf.keras.callbacks.Callback):

        def __init__(self, valid_data):
            super().__init__()
            self.validation_data = valid_data

        def on_train_batch_end(self, batch, logs=None):
            if (batch + 1) % 4 == 0:
                m = tf.keras.metrics.BinaryAccuracy()
                for idx, example in enumerate(self.validation_data):
                    m.update_state(example[1], self.model.predict(example[0]))
                    if idx >= 10:
                        break
                print(f"Validation accuracy: {m.result().numpy().mean()}")

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    save_every_num_batch = 10000
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True,
                                                             monitor='val_accuracy', mode='max',
                                                             save_freq=save_every_num_batch)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0
    )

    return [early_stopping, InterEpochValidation(valid_data)]


tf_train_dataset, tf_valid_dataset, tf_test_dataset = \
    generate_dataset(train_files_list=['tests/data/word2vec/ver2_parquet/*.parquet'],
                     valid_files_list=['tests/data/word2vec/ver2_parquet/*.parquet'],
                     test_files_list=['tests/data/word2vec/ver2_parquet/*.parquet'],
                    batch_size=2)


model = create_model()
# callbacks = get_callbacks('tests/data/word2vec/checkpoints', tf_valid_dataset)
# model.summary()

# x = next(iter(tf_train_dataset.take(1)))
# res = model(x[0])
# res[0]

model.fit(tf_train_dataset,
          steps_per_epoch=10,
          validation_data=tf_valid_dataset,
          epochs=1,
          callbacks=callbacks
          )

# tmp = next(iter(tf_train_dataset.take(1)))
# [tokenizer.decode(i) for i in tmp[0]['input_ids_0'][0].numpy().tolist()]
# loss, accuracy = model.evaluate(tf_test_dataset)
# logger.info("Evaluating on test set. Loss: {}, Accuracy: {}".format(loss, accuracy))
# model.save(f'{output_folder}/model1.h5')
