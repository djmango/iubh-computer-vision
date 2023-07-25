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
    
    for prediction, ground_truth in zip(predictions, ground_truths):
        predicted_labels = [segment.label for segment in prediction]
        ground_truth_labels = [segment.label for segment in ground_truth]
        
        true_positives += len(set(predicted_labels).intersection(ground_truth_labels))
        false_positives += len(set(predicted_labels) - set(ground_truth_labels))
        false_negatives += len(set(ground_truth_labels) - set(predicted_labels))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
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

