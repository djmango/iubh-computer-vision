from PIL import Image
import numpy as np
import requests
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# load Mask2Former fine-tuned on COCO panoptic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model: Mask2FormerForUniversalSegmentation = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic") # type: ignore

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for postprocessing
result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
predicted_panoptic_map = result["segmentation"]
segments_info = result["segments_info"]
predicted_panoptic_map_np = np.array(predicted_panoptic_map)

for segment in segments_info:
    assert model.config.id2label is not None
    # Obtain the id, label, score for each segment
    id = segment["id"]
    label_id = segment["label_id"]
    score = segment["score"]

    # Get the label name from the model config
    label_name = model.config.id2label[label_id]

    # Get the coordinates where the predicted panoptic map equals the segment's id
    coords = np.where(predicted_panoptic_map_np == id)
    
    # Compute bounding box as (xmin, ymin, xmax, ymax)
    ymin, xmin = np.min(coords, axis=1)
    ymax, xmax = np.max(coords, axis=1)
    bbox = [xmin, ymin, xmax, ymax]

    print(
        f"Detected {label_name} with confidence "
        f"{round(score, 3)} at location {bbox}"
    )

