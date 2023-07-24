from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from object_recognition.models import ObjectRecognition
from object_recognition.schemas import ObjectDetectionSegment

class Mask2FormerObjectRecognition(ObjectRecognition):
    def __init__(self, model_name):
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)

    def run_model(self, image: Image.Image) -> list[ObjectDetectionSegment]:
        inputs = self.processor(images=image, return_tensors="pt")

        # Run the model
        with torch.no_grad():
            outputs = self.model(**inputs) # type: ignore

        # Post process the outputs
        result = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_panoptic_map = result["segmentation"]
        segments_info = result["segments_info"]
        predicted_panoptic_map_np = np.array(predicted_panoptic_map)

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

        return result_segments

