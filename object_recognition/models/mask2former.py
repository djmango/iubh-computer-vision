import numpy as np
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from object_recognition.dataset import Dataset
from object_recognition.device import device
from object_recognition.models import ObjectRecognition
from object_recognition.schemas import ObjectDetectionSegment

class Mask2FormerObjectRecognition(ObjectRecognition):
    def __init__(self, model_name: str = "facebook/mask2former-swin-base-coco-panoptic"):
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
        self.model.to(device) # type: ignore

    def run_model(self, dataset: Dataset) -> list[list[ObjectDetectionSegment]]:
        """ Run the model on the images and return the results """
        results = self.run_model_on_batches(dataset)

        results_segments: list[list[ObjectDetectionSegment]] = []
        for i, batch in enumerate(results):
            print(f"Processing batch {i} with {self.model_name}")
            for result in batch:
                predicted_panoptic_map = result["segmentation"]
                segments_info = result["segments_info"]
                predicted_panoptic_map_np = np.array(predicted_panoptic_map.cpu())

                # Format the results in the required schema
                result_segments: list[ObjectDetectionSegment] = []
                for segment in segments_info:
                    # Obtain the id, label, score for each segment
                    id = segment["id"]
                    label_id = segment["label_id"]
                    score = segment["score"]

                    # Get the label name from the model config
                    label_name = self.model.config.id2label[label_id] # type: ignore

                    # Get the coordinates where the predicted panoptic map equals the segment's id
                    coords = np.where(predicted_panoptic_map_np == id)
                    
                    # Compute bounding box as (xmin, ymin, xmax, ymax)
                    ymin, xmin = np.min(coords, axis=1)
                    ymax, xmax = np.max(coords, axis=1)
                    bbox = [xmin, ymin, xmax, ymax]

                    result_segments.append(ObjectDetectionSegment(
                        label=label_name,
                        confidence=score,
                        bounding_box=bbox,
                    ))

                results_segments.append(result_segments)

        return results_segments

