import re
import sys
import os
import pandas
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            new_labels.append(label)

    return new_labels


tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-ner")
model = AutoModelForTokenClassification.from_pretrained("dicta-il/dictabert-ner")





enc = tokenizer.batch_encode_plus(["הלכתי אתמול  הביתה אחרי ההופעה"])
enc.tokens()
enc.word_ids()




