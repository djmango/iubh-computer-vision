from PIL import Image
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from object_recognition.models.abstract import ObjectRecognition
from object_recognition.schemas.segment import ObjectDetectionSegment


class DetrResnetObjectRecognition(ObjectRecognition):
    def __init__(self, model_name):
        self.model_name = model_name
        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model = DetrForObjectDetection.from_pretrained(self.model_name)

    def run_model(self, image: Image.Image) -> list[ObjectDetectionSegment]:
        inputs = self.processor(images=image, return_tensors="pt") # type: ignore
        outputs = self.model(**inputs) # type: ignore

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0] # type: ignore

        # Format results into ObjectDetectionSegment
        result_segments: list[ObjectDetectionSegment] = []
        for score, label, bounding_box in zip(results["scores"], results["labels"], results["boxes"]): # type: ignore
            label = self.model.config.id2label[label.item()] # type: ignore
            confidence = round(score.item(), 3)
            bounding_box = [round(i, 2) for i in bounding_box.tolist()]
            result_segments.append(ObjectDetectionSegment(
                label=label,
                confidence=confidence,
                bounding_box=bounding_box
            ))
        return result_segments
