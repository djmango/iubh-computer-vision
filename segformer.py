from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
from PIL import Image
import requests

processor: SegformerImageProcessor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512") # type: ignore
model: SegformerForSemanticSegmentation = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512") # type: ignore

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

# target_sizes = torch.tensor([image.size[::-1]])

target_sizes = [image.size[::-1]]
results = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]

seg_map = results[0]
unique_ids = seg_map.unique().tolist()

for class_id in unique_ids:
    assert model.config.id2label is not None
    class_name = model.config.id2label[class_id]
    print(f"Class ID: {class_id}, Class Name: {class_name}")

