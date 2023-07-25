from fuzzywuzzy import fuzz

from object_recognition.dataset import categories, limited_dataset
from object_recognition.models import (
    DetrResnetObjectRecognition,
    Mask2FormerObjectRecognition,
    YolosObjectRecognition,
)
from object_recognition.models.abstract import ObjectRecognition
from object_recognition.schemas.segment import ObjectDetectionSegment

# Create an instance of each class and process the image
models = {
    "Mask2Former": Mask2FormerObjectRecognition("facebook/mask2former-swin-base-coco-panoptic"),
    "Yolos": YolosObjectRecognition("hustvl/yolos-tiny"),
    "DetrResnet": DetrResnetObjectRecognition("facebook/detr-resnet-50"),
}

ground_truths: list[list[ObjectDetectionSegment]] = [ObjectDetectionSegment.from_dataset(x) for x in limited_dataset]

# for example in dataset:
#     labels = [categories.int2str(x) for x in example['objects']['category']]
#     print(labels)

def calculate_metrics(predictions: list[list[ObjectDetectionSegment]], ground_truths: list[list[ObjectDetectionSegment]]):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    similarity_threshold = 85  # You may need to adjust this threshold based on your requirements
    
    for prediction, ground_truth in zip(predictions, ground_truths):
        predicted_labels = [segment.label for segment in prediction]
        ground_truth_labels = [segment.label for segment in ground_truth]
        
        for predicted_label in predicted_labels:
            if any(fuzz.ratio(predicted_label, ground_truth_label) > similarity_threshold for ground_truth_label in ground_truth_labels):
                true_positives += 1
            else:
                false_positives += 1
                
        for ground_truth_label in ground_truth_labels:
            if not any(fuzz.ratio(ground_truth_label, predicted_label) > similarity_threshold for predicted_label in predicted_labels):
                false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


for name, model in models.items():
    model: ObjectRecognition
    import time
    start = time.perf_counter()
    
    images = [i['image'] for i in limited_dataset]

    print(f"Processing {len(images)} images with {name}")
    results = model.run_model(images)

    print(f"Time taken: {time.perf_counter() - start:.3f}s")
    precision, recall, f1_score = calculate_metrics(results, ground_truths)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1_score:.3f}")

