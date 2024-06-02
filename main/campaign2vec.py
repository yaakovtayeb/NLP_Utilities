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


def generate_dataset(train_files_list, valid_files_list, test_files_list, max_length=384, tokenizer=None, batch_size=1024):

    """
    Return tensorflow Dataset object for training
    :param List[String] train_files_list: list of parquet file paths to use for training
    :param List[String] valid_files_list: list of parquet file paths to use for validation
    :param List[String] test_files_list: list of parquet file paths to use for test
    :param int max_length: the size of the input id vector
    :param tokenizer: tokenizer object to use
    """

    dataset = load_dataset("parquet", data_files={'train': train_files_list, 'valid': valid_files_list,
                                                  'test': test_files_list})
    dataset = dataset.map(
        lambda examples: tokenizer(examples['text_0'], return_tensors='tf', padding=True, truncation=True,
                                   max_length=max_length), batched=True) \
        .rename_column("input_ids", "input_ids_0") \
        .rename_column("attention_mask", "attention_mask_0") \
        .map(lambda examples: tokenizer(examples['text_1'], return_tensors='tf', padding=True, truncation=True,
                                        max_length=max_length), batched=True) \
        .rename_column("input_ids", "input_ids_1") \
        .rename_column("attention_mask", "attention_mask_1") \
        .select_columns(["input_ids_0", "attention_mask_0", "input_ids_1", "attention_mask_1", "label"])

    tf_train_dataset = dataset['train'].to_tf_dataset(columns=["input_ids_0", "attention_mask_0", "input_ids_1", "attention_mask_1"],
                                                     label_cols='label',
                                                     batch_size=batch_size,
                                                     shuffle=True)
    tf_valid_dataset = dataset['valid'].to_tf_dataset(
        columns=["input_ids_0", "attention_mask_0", "input_ids_1", "attention_mask_1"],
        label_cols='label',
        batch_size=batch_size,
        shuffle=True)
    tf_test_dataset = dataset['test'].to_tf_dataset(columns=["input_ids_0", "attention_mask_0", "input_ids_1", "attention_mask_1"],
                                                     label_cols='label',
                                                     batch_size=batch_size,
                                                     shuffle=True)

    return tf_train_dataset, tf_valid_dataset, tf_test_dataset


def create_model(orig_model, max_length=384):
    input_ids_0 = tf.keras.Input(shape=(max_length,), name='input_ids_0', dtype='int64')
    mask_0 = tf.keras.Input(shape=(max_length,), name='attention_mask_0', dtype='int64')
    input_ids_1 = tf.keras.Input(shape=(max_length,), name='input_ids_1', dtype='int64')
    mask_1 = tf.keras.Input(shape=(max_length,), name='attention_mask_1', dtype='int64')
    embeddings_0 = orig_model(input_ids_0, mask_0)[1]
    embeddings_1 = orig_model(input_ids_1, mask_1)[1]
    collaborative_layer = tf.keras.layers.Dense(100, activation='relu', name='collaborative_layer')
    co_text0 = collaborative_layer(embeddings_0)
    co_text1 = collaborative_layer(embeddings_1)
    # x = tf.keras.layers.Concatenate()([co_text0, co_text1])
    merged = tf.keras.layers.Dot(name='dot', normalize=True, axes=1)([co_text0, co_text1])
    merged = tf.keras.layers.Reshape(target_shape=[1])(merged)
    y = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(merged)
    model = tf.keras.Model(inputs=[input_ids_0, mask_0, input_ids_1, mask_1], outputs=y)
    model.layers[4].trainable = False
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


model_id = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_id)
orig_model = TFAutoModel.from_pretrained(model_id)

tf_train_dataset, tf_valid_dataset, tf_test_dataset = \
    generate_dataset(train_files_list=['tests/data/word2vec/training_data_parquet/*.parquet'],
                     valid_files_list=['tests/data/word2vec/validation_data_parquet/*.parquet'],
                     test_files_list=['tests/data/word2vec/validation_data_parquet/*.parquet'],
                     tokenizer=tokenizer, batch_size=3)

model = create_model(orig_model)
callbacks = get_callbacks('tests/data/word2vec/checkpoints', tf_valid_dataset)
# model.summary()

model.fit(tf_train_dataset,
          steps_per_epoch=20,
          validation_data=tf_valid_dataset,
          epochs=1,
          callbacks=callbacks
          )

# tmp = next(iter(tf_train_dataset.take(1)))
# [tokenizer.decode(i) for i in tmp[0]['input_ids_0'][0].numpy().tolist()]
# loss, accuracy = model.evaluate(tf_test_dataset)
# logger.info("Evaluating on test set. Loss: {}, Accuracy: {}".format(loss, accuracy))
# model.save(f'{output_folder}/model1.h5')

