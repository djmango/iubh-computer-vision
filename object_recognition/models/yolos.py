from typing import Iterable

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from object_recognition.device import device
from object_recognition.models.abstract import ObjectRecognition
from object_recognition.schemas.segment import ObjectDetectionSegment

class YolosObjectRecognition(ObjectRecognition):
    def __init__(self, model_name: str = "hustvl/yolos-tiny"):
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
        self.model.to(device) # type: ignore

    def run_model(self, images: Iterable[Image.Image]) -> list[list[ObjectDetectionSegment]]:
        """ Run the model on the images and return the results """
        results = self.run_model_on_batches(images)

        results_segments: list[list[ObjectDetectionSegment]] = []
        for i, batch in enumerate(results):
            print(f"Processing batch {i} with {self.model_name}")
            for result in batch:
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

