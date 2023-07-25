from itertools import islice

from datasets import load_dataset

limit = 10

dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)
categories = dataset.features["objects"].feature["category"] # type: ignore
limited_dataset = list(islice(dataset, limit))

