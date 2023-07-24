import requests
from PIL import Image
from object_recognition.models import Mask2FormerObjectRecognition

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Create an instance of each class and process the image
models = {
    "Mask2Former": Mask2FormerObjectRecognition("facebook/mask2former-swin-base-coco-panoptic"),
    # "Yolos": YolosObjectRecognition("hustvl/yolos-tiny"),
    # "DetrResnet": DetrResnetObjectRecognition("facebook/detr-resnet-50"),
}

for name, model in models.items():
    print(f"Processing image with {name}")
    results = model.run_model(image)
    model.print_results(results)
    model.display_results(results, image)

