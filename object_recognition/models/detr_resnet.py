from PIL import Image
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from object_recognition.device import device
from object_recognition.models.abstract import ObjectRecognition
from object_recognition.schemas.segment import ObjectDetectionSegment


class DetrResnetObjectRecognition(ObjectRecognition):
    def __init__(self, model_name):
        self.model_name = model_name
        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model = DetrForObjectDetection.from_pretrained(self.model_name)
        self.model.to(device) # type: ignore

    def run_model(self, images: list[Image.Image]) -> list[list[ObjectDetectionSegment]]:
        # Convert images to float tensors and normalize if necessary
        inputs = self.processor(images=images, return_tensors="pt") # type: ignore

        # Move the tensors to the device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs) # type: ignore

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = [image.size[::-1] for image in images]
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9) # type: ignore
        results_segments: list[list[ObjectDetectionSegment]] = []

        for result in results:
            # Format results into ObjectDetectionSegment
            result_segments: list[ObjectDetectionSegment] = []
            for score, label, bounding_box in zip(result["scores"], result["labels"], result["boxes"]): # type: ignore
                label = self.model.config.id2label[label.item()] # type: ignore
                confidence = round(score.item(), 3)
                bounding_box = [round(i, 2) for i in bounding_box.tolist()]
                result_segments.append(ObjectDetectionSegment(
                    label=label,
                    confidence=confidence,
                    bounding_box=bounding_box
                ))
            results_segments.append(result_segments)

        return results_segments

