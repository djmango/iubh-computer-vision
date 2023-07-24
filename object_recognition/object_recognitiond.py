from abc import ABC, abstractmethod

from PIL import Image
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    DetrForObjectDetection,
    DetrImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


class YolosObjectRecognition(ObjectRecognition):
    def __init__(self, model_name):
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)

    def detect_objects(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def post_process(self, outputs, image):
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        return results

    def print_results(self, results, confidence_threshold: float = 0.9):
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= confidence_threshold:
                print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

    def display_results(self, results, image, confidence_threshold: float = 0.9, box_color: str = 'r'):
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= confidence_threshold:
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=box_color, facecolor='none')
                ax.add_patch(rect)
        plt.show()


class DetrResnetObjectRecognition(ObjectRecognition):
    def __init__(self, model_name):
        self.model_name = model_name
        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model = DetrForObjectDetection.from_pretrained(self.model_name)

    def detect_objects(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs) # type: ignore
        return outputs

    def post_process(self, outputs, image):
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0] # type: ignore
        return results

    def print_results(self, results, confidence_threshold: float = 0.9):
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= confidence_threshold:
                print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

    def display_results(self, results, image, confidence_threshold: float = 0.9, box_color: str = 'r'):
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= confidence_threshold:
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=box_color, facecolor='none')
                ax.add_patch(rect)
        plt.show()
