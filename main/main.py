import os
import glob
import logging
import numpy as np
import pandas as pd
from tensorflow.python.framework.ops import disable_eager_execution
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from datasets import load_dataset
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained("bert-base-cased")

devices = tf.config.list_physical_devices()
tf.debugging.set_log_device_placement(True)

# disable_eager_execution()

logging.basicConfig(format='%(name)s | %(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger('Bert Classifier')
logger.setLevel(logging.INFO)


def generate_dataset(train_file_list, test_file_list, text_field='text', max_length=20, tokenizer=None, batch_size=1024):

    """
    Return tensorflow Dataset object for training
    :param List[String] train_file_list:list of parquet file paths to use for training
    :param List[String] test_file_list:list of parquet file paths to use for test
    :param string text_field: name of colum to tokenize
    :param int max_length: the size of the input id vector
    :param tokenizer: tokenizer object to use
    """

    dataset = load_dataset("parquet", data_files={'train': train_file_list, 'test': test_file_list})
    dataset = dataset.map(lambda examples: tokenizer(examples[text_field], return_tensors='tf', truncation=True,
                                                     padding='max_length', max_length=max_length), batched=True)
    tf_train_dataset = dataset['train'].to_tf_dataset(columns=['input_ids', 'attention_mask'],
                                                     label_cols='label',
                                                     batch_size=batch_size,
                                                     shuffle=True)
    tf_test_dataset = dataset['test'].to_tf_dataset(columns=['input_ids', 'attention_mask'],
                                                     label_cols='label',
                                                     batch_size=batch_size,
                                                     shuffle=True)

    return tf_train_dataset, tf_test_dataset


def create_model(lr_scheduler):
    input_ids = tf.keras.Input(shape=(20,), name='input_ids', dtype='int64')
    mask = tf.keras.Input(shape=(20,), name='attention_mask', dtype='int64')
    embeddings = bert.bert(input_ids, mask)[1]
    x = tf.keras.layers.Dense(100, activation='relu')(embeddings)
    y = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    model.layers[2].trainable = False
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
        metrics=['accuracy'],
    )
    model.layers[2].trainable = False
    logger.info(model.summary())
    return model


tf_train_dataset, tf_test_dataset = generate_dataset(['tests/data/training_data_parquet/file1.parquet','tests/data/training_data_parquet/file2.parquet'],
                                                     ['tests/data/training_data_parquet/file1.parquet'],
                                                     text_field='text', max_length=20, tokenizer=tokenizer)

EPOCHS = 1
num_train_steps = len(tf_train_dataset) * EPOCHS
lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)

bert_model = create_model(lr_scheduler)
bert_model.fit(tf_train_dataset,
               validation_data=tf_test_dataset,
               epochs=EPOCHS,
               )

# next(iter(training_data.map(lambda x, y: y).take(1)))
# next(iter(tf_train_dataset.take(1)))
# res = model.predict(tf_train_dataset.take(1))

# x = ({'input_ids': tf.convert_to_tensor(dataset['test'][0:1]['input_ids']),
#       'attention_mask': tf.convert_to_tensor(dataset['test'][0:1]['attention_mask'])
#       })
# model.predict(x)

# encoded_input = tokenizer(text,  return_tensors='tf', truncation=True, padding='max_length', max_length=20)
# tokenizer.decode(encoded_input['input_ids'][0])

