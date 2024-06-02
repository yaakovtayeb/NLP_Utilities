import os
import re
import glob
import logging
import numpy as np
import pandas as pd
from tensorflow.python.framework.ops import disable_eager_execution
from datetime import datetime
import tensorflow as tf
from datasets import load_dataset
from transformers import TFAutoModelWithLMHead
from transformers import BertTokenizer, TFBertModel, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import create_optimizer

devices = tf.config.list_physical_devices()
tf.debugging.set_log_device_placement(True)
logger = logging.getLogger('Bert Classifier')
logging.basicConfig(format='%(name)s | %(asctime)s %(levelname)s:%(message)s')
logger.setLevel(logging.INFO)


def align_input(parquet_filename, col_name='text', chunk_size=128):
    data = pd.read_parquet(parquet_filename)
    total_text = ' '.join(data[col_name].tolist())
    total_text = re.sub(r'<[^<]*>', '', total_text).split(' ')

    row = []
    for i in range(0, len(total_text), chunk_size):
        row += [' '.join(total_text[i:i + chunk_size])]
    new_data = pd.DataFrame(row, columns=[col_name])
    return new_data


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="np")
    token_logits = model(**inputs).logits
    mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()

    for token in top_5_tokens:
        print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
print(tokenizer.is_fast)
model = TFAutoModelWithLMHead.from_pretrained("distilbert-base-cased")
model.summary()

predict("This is a great [MASK].", tokenizer, model)

# For retraining MASK task, it is recommended to repartition the example to the same size without truncation.
# while this can be done in datasets objects, I chose to do it in pandas dataframe

# Load dataset
# train_df = align_input('tests/data/imdb/imdb-train.parquet')
# train_df.to_parquet('tests/data/imdb/imdb-train-128.parquet', index=False, compression='snappy')
# test_df = align_input('tests/data/imdb/imdb-test.parquet')
# test_df.to_parquet('tests/data/imdb/imdb-test-128.parquet', index=False, compression='snappy')


def generate_dataset(train_files, test_path, max_length=150, batch_size=1000):
    def tokenize_function(examples):
        result = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=max_length)
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    dataset = load_dataset("parquet", data_files={'train': train_files, 'test': test_path})
    dataset = dataset.map(tokenize_function, batched=True) \
        .map(lambda examples: {'labels': examples['input_ids'].copy()}, batched=True,
             remove_columns=['text', 'word_ids'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    tf_train_dataset = dataset['train'].to_tf_dataset(columns=['input_ids', 'attention_mask'],
                                                     label_cols='label',
                                                     collate_fn=data_collator,
                                                     batch_size=batch_size,
                                                     shuffle=True)
    tf_test_dataset = dataset['test'].to_tf_dataset(columns=['input_ids', 'attention_mask'],
                                                     label_cols='label',
                                                     collate_fn=data_collator,
                                                     batch_size=batch_size,
                                                     shuffle=True)

    return dataset, tf_train_dataset, tf_test_dataset


dataset, training_data, testing_data = generate_dataset(['tests/data/imdb/imdb-train-128.parquet'], ['tests/data/imdb/imdb-test-128.parquet'])

# dataset['train'][0]
# next(iter(testing_data.take(1)))

num_train_steps = dataset['train'].num_rows // 1000
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            )

model.fit(training_data, validation_data=testing_data, epochs=1)


# text = "The quick brown [MASK] jumps over the lazy dog."
# input = tokenizer.encode(text, return_tensors="tf")
# mask_token_index = tf.where(input == tokenizer.mask_token_id)[0, 1]
# token_logits = bert(input)[0]
# mask_token_logits = token_logits[0, mask_token_index, :]
# top_5_tokens = tf.math.top_k(mask_token_logits, 5).indices.numpy()
#
# for token in top_5_tokens:
#     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
#
