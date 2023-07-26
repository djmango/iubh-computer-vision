from abc import ABC, abstractmethod
import random
import typing

import torch
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from object_recognition.device import device
if typing.TYPE_CHECKING:
    from object_recognition.schemas import ObjectDetectionSegment

def normalize_image(image: Image.Image) -> Image.Image:
    """Normalize a PIL Image."""

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert the image to a NumPy array
    array = np.array(image)

    # If the image is grayscale, add a channel dimension
    if array.ndim == 2:
        array = array[..., np.newaxis]

    # Normalize the array
    array = array / 255.0

    # Convert the normalized array back to a PIL Image
    normalized_image = Image.fromarray((array * 255).astype(np.uint8))

    return normalized_image

class ObjectRecognition(ABC):
    """ Abstruct model for my ObjectRecognition classes.
    These are wrappers around pretrained Transformers Image Segmentation and Object Recognition models
    They have basic functionality with unified I/O for ease of usage and testing
    """

    @abstractmethod
    def __init__(self, model_name):
        self.model_name = model_name
        self.processor = None
        self.model = None

    def get_inputs(self, images: list[Image.Image]):
        """ Get the inputs for the model from the images, on the device """
        assert self.processor is not None, "Processor is not initialized"

        # Convert images to float tensors and normalize if necessary
        images_normalized = [normalize_image(image) for image in images]
        inputs = self.processor(images=images_normalized, return_tensors="pt")

        # Move the tensors to the device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()} # type: ignore
        return inputs

    def run_model_on_batches(self, images: list[Image.Image], batch_size: int = 128) -> list:
        assert self.model is not None, "Model is not initialized"
        assert self.processor is not None, "Processor is not initialized"
        results = []
        
        # Split images into batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            inputs = self.get_inputs(batch_images)

            # Run the model
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post process the outputs
            target_sizes = [image.size[::-1] for image in batch_images]

            if self.model_name == "facebook/mask2former-swin-base-coco-panoptic":
                batch_results = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)
            else:
                batch_results = self.processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)

            # Save batch results
            results.extend(batch_results)

        return results

    @abstractmethod
    def run_model(self, images: list[Image.Image]) -> list[list['ObjectDetectionSegment']]:
        pass

    def print_results(self, results: list['ObjectDetectionSegment'], confidence_threshold: float = 0.9):
        for result in results:
            if result.confidence >= confidence_threshold:
                print(f"Detected {result.label} with confidence {result.confidence:.3f} at location {result.bounding_box}")

    def display_results(self, results: list['ObjectDetectionSegment'], image: Image.Image, confidence_threshold: float = 0.9):
        """Display image with bounding boxes for detected objects"""
        
        # Create a figure and axis for the image
        fig, ax = plt.subplots(1)
        ax.imshow(image) # type: ignore
        
        # Get unique labels to generate distinct colors
        unique_labels = set([result.label for result in results])
        colors = {label: (random.random(), random.random(), random.random()) for label in unique_labels}
        
        for result in results:
            if result.confidence >= confidence_threshold:
                # Draw bounding box
                box = result.bounding_box
                width = box[2] - box[0]
                height = box[3] - box[1]
                rect = Rectangle((box[0], box[1]), width, height, linewidth=1, edgecolor=colors[result.label], facecolor='none')
                ax.add_patch(rect) # type: ignore
                
                # Add label and confidence score
                label = f"{result.label}: {result.confidence:.2f}"
                plt.text(box[0], box[1], label, color=colors[result.label])
        
        plt.show()

