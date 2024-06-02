from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
text = "Replace me by any text you'd like."

encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
output['last_hidden_state']