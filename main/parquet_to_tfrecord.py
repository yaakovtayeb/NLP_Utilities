import os
import logging
import glob
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
from tensorflow.python.framework.ops import disable_eager_execution
import pandas as pd

logging.basicConfig(format='%(name)s | %(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger('Generate TFRecord')
logger.setLevel(logging.INFO)
#
devices = tf.config.list_physical_devices()
# tf.debugging.set_log_device_placement(True)

# disable_eager_execution()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(text, label):

    """
    Creates a tf.train.Example message ready to be written to a file.
    example = serialize_example(1002, 4, 1)
    show the decoded data: tf.train.Example.FromString(example)
    """

    feature = {
      'text': _bytes_feature(text),
      'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(f0,f1):
    """
    us tf.functions to make the function graph-able
    :param f0: company_idx
    :param f1: user_idx
    :param f2: label
    :return:
    """
    tf_string = tf.py_function(serialize_example, (f0, f1), tf.string)
    return tf.reshape(tf_string, ())


def divide_and_save(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    file_list = glob.glob(input_folder)
    for idx, f in enumerate(file_list):
        parquet_file = pq.ParquetFile(f)
        i = 0
        filename = f'{output_folder}{f.split("/")[-1]}_{{i}}.tfrecord'
        for batch in parquet_file.iter_batches(batch_size=10_000):
            pd_batch = batch.to_pandas()

            # example = serialize_example(pd_batch['company_idx'].tolist(), pd_batch['user_idx'].tolist(),
            #                             pd_batch['label'].tolist())
            # with tf.io.TFRecordWriter(filename.format(**{'i': i})) as writer:
            #     writer.write(example)

            features_dataset = tf.data.Dataset.from_tensor_slices((pd_batch['text'].values,
                                                                   pd_batch['label'].values))
            serialized_features_dataset = features_dataset.map(tf_serialize_example)

            writer = tf.data.experimental.TFRecordWriter(filename.format(**{'i': i}))
            writer.write(serialized_features_dataset)
            i += 1
        logger.info("finished single file")
    logger.info('Done')


print(f"Working FolderL: {os.getcwd()}")
divide_and_save(input_folder='tests/data/training_data_parquet/*.parquet',
               output_folder='tests/data/training_data_tf_records/')
#
# # load a file:
# filenames = ['/Users/ytayeb1/Projects/Bert_Veggies/tests/data/training_data_tf_records/file1.parquet_0.tfrecord']
# raw_dataset = tf.data.TFRecordDataset(filenames)
# feature_description = {
#     'text': tf.io.FixedLenFeature([], tf.string),
#     'label': tf.io.FixedLenFeature([], tf.int64)
# }
#
#
# def _parse_function(example_proto):
#   # Parse the input `tf.train.Example` proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, feature_description)
#
#
# parsed_dataset = raw_dataset.map(_parse_function)
# data = next(iter(parsed_dataset.batch(10).take(1)))


# next(iter(serialized_features_dataset.take(1)))

# tuple(map(tuple, pd_batch.head(5).values)
