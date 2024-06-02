from transformers import AutoTokenizer, TFRobertaForQuestionAnswering
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("ydshieh/roberta-base-squad2")
model = TFRobertaForQuestionAnswering.from_pretrained("ydshieh/roberta-base-squad2")

question, text = "What color is the sky?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="tf")
print(f"Input Keys: {inputs.keys()}")
outputs = model(**inputs)

answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
answer_probability = tf.math.softmax(outputs['start_logits'], axis=-1)[0][answer_start_index]

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)