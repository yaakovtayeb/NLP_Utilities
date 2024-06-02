import logging
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import DataCollatorForTokenClassification
import tensorflow as tf
import evaluate
import numpy as np

logging.basicConfig(format='%(name)s | %(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger('multiclass classification')
logger.setLevel(logging.INFO)

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
logger.info(f"Tokenzier fast: {tokenizer.is_fast}")
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# input_ids
# token_type_ids
# attention_mask
