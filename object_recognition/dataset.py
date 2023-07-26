from itertools import islice

from datasets import IterableDataset, load_dataset

limit = 10

class Dataset():
    def __init__(self, limit):
        self.limit = limit
        self.dataset = None
        self.categories = None

    def get_dataset(self) -> IterableDataset:
        if not self.dataset:
            self.dataset = load_dataset("detection-datasets/coco", split="val", streaming=True, verification_mode="no_checks", save_infos=True, keep_in_memory=False)

        assert isinstance(self.dataset, IterableDataset), "Dataset is not iterable"
        return self.dataset
    
    def get_categories(self):
        if not self.categories:
            self.categories = self.get_dataset().features["objects"].feature["category"] # type: ignore

        return self.categories
    
    def get_limited_dataset(self):
        pass
    
    @property
    def limited_dataset(self):
        return islice(self.get_dataset(), self.limit)

dataset = Dataset(limit=limit)
