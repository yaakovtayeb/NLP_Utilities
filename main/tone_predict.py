# https://huggingface.co/alexandrainst/da-sentiment-base

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
model = TFAutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

devices = tf.config.list_physical_devices()
tf.debugging.set_log_device_placement(True)


def generate_results(text):
    input = tokenizer.encode(text, return_tensors="tf")
    predictions = tf.nn.softmax(model(input)['logits']).numpy()[0].tolist()
    pred_item = tf.argmax(predictions).numpy()
    print(f"Results: {labels[pred_item]}: {predictions[pred_item]}")
    # full results:
    for i in list(zip(labels, predictions)):
        print(f"{i[0]}: {i[1]}")
    return labels[pred_item], predictions[pred_item]


generate_results("I had a great time, can't wait for next time")

