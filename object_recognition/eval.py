from datasets import load_dataset
from itertools import islice

from object_recognition.models import (
    DetrResnetObjectRecognition,
    Mask2FormerObjectRecognition,
    YolosObjectRecognition,
)
from object_recognition.models.abstract import ObjectRecognition

# Create an instance of each class and process the image
models = {
    "Mask2Former": Mask2FormerObjectRecognition("facebook/mask2former-swin-base-coco-panoptic"),
    "Yolos": YolosObjectRecognition("hustvl/yolos-tiny"),
    "DetrResnet": DetrResnetObjectRecognition("facebook/detr-resnet-50"),
}

dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)

for name, model in models.items():
    model: ObjectRecognition
    import time
    start = time.perf_counter()
    
    limit = 100
    limited_dataset = islice(dataset, limit)
    images = [i['image'] for i in limited_dataset]
    # images = [image.convert("RGB").resize((1920, 1080)) for image in images]

    images = images[:3]

    print(f"Processing {len(images)} images with {name}")
    results = model.run_model(images)

    print(f"Time taken: {time.perf_counter() - start:.3f}s")
    model.print_results(results[0])
    model.display_results(results[0], images[0])

