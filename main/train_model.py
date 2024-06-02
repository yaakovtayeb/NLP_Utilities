import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from datetime import datetime

devices = tf.config.list_physical_devices()
# tf.debugging.set_log_device_placement(True)

# disable_eager_execution()


def generate_dataset(folder_path, batch_size=3):

    def _parse_function(example_proto):
        # return tf.io.parsesingle__example(example_proto, feature_description)
        x = tf.io.parse_example(example_proto, feature_description)
        return (x['company_idx'], x['user_idx']), x['label']

    file_list = glob.glob(folder_path)
    feature_description = {
        'company_idx': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'user_idx': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    # next(iter(tf.data.TFRecordDataset(file_list).batch(batch_size).map(_parse_function)))
    return tf.data.TFRecordDataset(file_list).batch(batch_size).prefetch(batch_size*4).map(_parse_function).repeat()


def get_callbacks():
    log_dir = f'{os.getcwd()}/logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    checkpoint_dir = f"{os.getcwd()}/embedding_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    # save_every_num_batch = 10000
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=False,
                                                             save_freq='epoch')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', profile_batch=1,
                                                          histogram_freq=0, write_images=False)
    callbacks = [tensorboard_callback]

    return callbacks


def build_model(item_embedding=128, user_embedding=128, item_input_dim=0, user_input_dim=0, loss=tf.keras.losses.binary_crossentropy):
    item_input = tf.keras.layers.Input(name='item_input', shape=(1,))
    user_input = tf.keras.layers.Input(name='user_input', shape=(1,))
    item_emb = tf.keras.layers.Embedding(name='company_embedding',
                                            input_dim=item_input_dim,
                                            output_dim=item_embedding)(item_input)
    item_emb = tf.keras.layers.Flatten()(item_emb)
    user_emb = tf.keras.layers.Embedding(name='user_embedding',
                                          input_dim=user_input_dim,
                                          output_dim=user_embedding)(user_input)
    user_emb = tf.keras.layers.Flatten()(user_emb)
    merged = tf.keras.layers.Dot(name='dot', normalize=True, axes=1)([user_emb, item_emb])
    merged = tf.keras.layers.Reshape(target_shape=[1])(merged)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    model = tf.keras.Model(inputs=(item_input, user_input), outputs=x)

    model.compile(
        loss=loss,
        optimizer='adam',
        metrics=['accuracy'],
    )

    return model


print(f"Working folder: {os.getcwd()}")

training_data = generate_dataset('data/tf_records/training_data_parquet/*.tfrecord', batch_size=2048)
validation_data = generate_dataset('data/tf_records/validation_data_parquet/*.tfrecord', batch_size=2048)
test_data = generate_dataset('data/tf_records/test_data_parquet/*.tfrecord', batch_size=2048)

# x = iter(test_data)
# x.next()
# model(x.next()[0])
# with tf.device('/GPU:0'):
model = tf.keras.models.load_model('models/model1.h5')
# model.summary()

# model = build_model(item_input_dim=220414, user_input_dim=10001, item_embedding=32, user_embedding=32)
EPOCHS = 5
callbacks = get_callbacks()
r = model.fit(training_data,
              validation_data=validation_data,
              steps_per_epoch=(618_378 // 2048) + 1,
              validation_steps=(54_396 // 2048) + 1,
              epochs=EPOCHS,
              callbacks=callbacks
              )

model.save('models/model1.h5')
model.evaluate(test_data.take(25))

# x.next()
# model.predict([item_input, user])
# model.predict([np.array([92408]), np.array([913])])

weights = model.layers[3].get_weights()[0]
weights_pd = pd.DataFrame(weights, columns=[f"emb_{i}" for i in range(32)])
weights_pd.reset_index().to_csv('weights/w1.csv', index=False)