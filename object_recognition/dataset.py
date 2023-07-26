from itertools import islice

from datasets import load_dataset

limit = 1000

class Dataset():
    def __init__(self, limit):
        self.limit = limit
        self.dataset = None
        self.categories = None
        self.limited_dataset = None

    def get_dataset(self):
        if not self.dataset:
            self.dataset = load_dataset("detection-datasets/coco", split="val", streaming=True, verification_mode="no_checks", save_infos=True)

        return self.dataset
    
    def get_categories(self):
        if not self.categories:
            self.categories = self.get_dataset().features["objects"].feature["category"] # type: ignore

        return self.categories
    
    def get_limited_dataset(self):
        if not self.limited_dataset:
            self.limited_dataset = list(islice(self.get_dataset(), self.limit))

        return self.limited_dataset

dataset = Dataset(limit=limit)
