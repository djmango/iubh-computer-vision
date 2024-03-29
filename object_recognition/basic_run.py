from PIL import Image
import requests

from object_recognition.models import (
    DetrResnetObjectRecognition,
    Mask2FormerObjectRecognition,
    YolosObjectRecognition,
)
from object_recognition.models.abstract import ObjectRecognition

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Create an instance of each class and process the image
models = {
    "Mask2Former": Mask2FormerObjectRecognition("facebook/mask2former-swin-base-coco-panoptic"),
    "Yolos": YolosObjectRecognition("hustvl/yolos-tiny"),
    "DetrResnet": DetrResnetObjectRecognition("facebook/detr-resnet-50"),
}

for name, model in models.items():
    model: ObjectRecognition
    import time
    start = time.perf_counter()
    print(f"Processing image with {name}")
    results = model.run_model([image])
    print(f"Time taken: {time.perf_counter() - start:.2f}s")
    model.print_results(results[0])
    model.display_results(results[0], image)

