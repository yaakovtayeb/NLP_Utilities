import logging
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
import tensorflow as tf
import evaluate
import numpy as np

logging.basicConfig(format='%(name)s | %(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger('tokens classification')
logger.setLevel(logging.INFO)


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
logger.info(f"Tokenzier fast: {tokenizer.is_fast}")

raw_datasets = load_dataset("conll2003")
label_names = raw_datasets['train'].features["ner_tags"].feature.names
logger.info(f"Label names: {label_names}")
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)

tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)

# next(iter(tf_eval_dataset.take(1)))

model = TFAutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
logger.info(f"Model MultiClass Num: {model.config.num_labels}")
model.layers[0].trainable = False
model.summary()

model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'],
    )

# res = model.predict([[101,  1650,   188,  8928,  3491,  5327,  1125,  1680, 10601,
#          1111,  1210,   119,   102,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0]])
# res[0][0]

model.fit(tf_train_dataset,
          validation_data=tf_eval_dataset,
          epochs=1
          )

# Evaluate model
all_predictions = []
all_labels = []
for batch in tf_eval_dataset:
    logits = model.predict_on_batch(batch)["logits"]
    labels = batch["labels"]
    predictions = np.argmax(logits, axis=-1)
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(label_names[predicted_idx])
            all_labels.append(label_names[label_idx])
metric.compute(predictions=[all_predictions], references=[all_labels])




#
# inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
# inputs.tokens()
# inputs.word_ids()
# align_labels_with_tokens(raw_datasets["train"][0]["ner_tags"],
#                          tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True).word_ids())
#
# tokenized_datasets['train'][0]