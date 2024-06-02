from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# take datasets from https://huggingface.co/datasets/amazon_reviews_multi, filter and do some basic filtering
english_dataset = load_dataset("amazon_reviews_multi", "en")
english_dataset = english_dataset.filter(lambda example: example['product_category'] == 'book' or example["product_category"] == "digital_ebook_purchase")
english_dataset['train'] = english_dataset['train'].remove_columns(['review_id', 'product_id', 'reviewer_id', 'stars', 'language', 'product_category'])
english_dataset['test'] = english_dataset['test'].remove_columns(['review_id', 'product_id', 'reviewer_id', 'stars', 'language', 'product_category'])
english_dataset['validation'] = english_dataset['validation'].remove_columns(['review_id', 'product_id', 'reviewer_id', 'stars', 'language', 'product_category'])
english_dataset = english_dataset.filter(lambda x: len(x["review_title"].split()) > 2)

# load summarization model and tokenizer
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

summary = '''The Barbie movie is not what you think it is. Droves of audience members may be familiar with the iconic doll, but the film adaptation is an unexpected adventure. As stereotypical Barbie, Margot Robbie brings the blonde beauty to life living in the paradise that is Barbie Land. Every day is perfect, as was the day before, and all the days to come. The only problem is Barbie’s sudden and inexplicable feeling of anxiety, and thoughts of her impending doom.
It’s a high concept for a movie about a doll, and it’s only the film’s basic premise. Several times in the movie you’re faced with ideas that feel like the kind of thing that wouldn’t make it past an initial pitch. As Barbie ventures into the real world, she’s faced with misogynistic comments, and a lack of agency. An opposite to the reality she’s known where she and her Barbie counterparts make up the Supreme Court and the Presidency of Barbie Land.
Barbie may have centre stage, but Ken has plenty of material himself. Ryan Gosling commits to playing the ultimate himbo. His worship of Barbie is boundless, and is more like a satellite in her orbit than a planet to himself. The film deals with the psyche of a walking talking accessory, brashly taking on ideas he was never meant to encounter.
Buzz walked so Ken could frolic
With Barbie and Ken suffering existential crises, the film’s subject matter seems at odds with Barbie’s signature pink and pastel perfection. The brilliance of the movie is how it takes that identity and engages with it critically. Anything one might say about Barbie is dealt with, from her impact on beauty standards, to the insistence on exceptionalism as a matter of fact. Simply put, Barbie gets taken to task, in a way legacy icons seldom do.
As challenging as the movie is to its main character, the film is undoubtedly a celebration of her history. Barbie does a deep dive on the several incarnations of the toy, even referencing obscure versions that were quickly discontinued. The sense of reverence for the source material is no more evident than in the brightly coloured set design of Barbie Land. In a world where blockbuster films are surrounded by digital landscapes, Barbie’s devotion to practical production is outstanding. 
Hello Academy? I think you have a winner. Yes it’s Barbie
Barbie is ultimately a film about an idea coming to terms with its effect on the world. It presents an ideal and quickly points out its limitations. The film engages with notions of femininity and masculinity, showcasing the harm in imposing them on people that don’t quite fit into a neatly packaged box. At its core the movie celebrates being just plain ordinary, all the while being one of the weirdest, funniest, most spectacular, star studded, and emotional experiences you could have in the cinema. '''

summary = """The Barbie movie is not what you think it is. Droves of audience members may be familiar with the iconic doll"""
input_ids = tokenizer(summary, max_length=1024, return_tensors="pt")

summary_ids = model.generate(input_ids['input_ids'], max_length=1024, num_beams=5)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
