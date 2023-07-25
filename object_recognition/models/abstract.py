from abc import ABC, abstractmethod
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import random

from PIL import Image
import typing
if typing.TYPE_CHECKING:
    from object_recognition.schemas import ObjectDetectionSegment

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

