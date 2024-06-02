from datasets import Dataset, interleave_datasets

# Interleave Datasets (Down / Up sampling)
probabilities = [0.3, 0.5, 0.2]
dataset1 = Dataset.from_dict({"a": list(range(10))})     # size = 3
dataset2 = Dataset.from_dict({"a": list(range(10, 99)) * 10}).shuffle()  # size = 890
dataset = interleave_datasets([dataset1, dataset2], probabilities=[0.9, 0.1], stopping_strategy="all_exhausted")


