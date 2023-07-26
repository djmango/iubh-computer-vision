from datetime import datetime
from pathlib import Path
import platform
import time

from fuzzywuzzy import fuzz
import pandas as pd

from object_recognition.dataset import dataset
import psutil
from object_recognition.models import (
    DetrResnetObjectRecognition,
    Mask2FormerObjectRecognition,
    YolosObjectRecognition,
)
from object_recognition.models.abstract import ObjectRecognition
from object_recognition.schemas.segment import ObjectDetectionSegment
from object_recognition.system_monitor import SystemMonitor

HERE = Path(__file__).parent
EVAL_FOLDER = HERE.parent / "eval"

# Create an instance of each class and process the image
models = {
    "Mask2Former": Mask2FormerObjectRecognition(),
    "Yolos": YolosObjectRecognition(),
    "DetrResnet": DetrResnetObjectRecognition(),
}

ground_truths: list[list[ObjectDetectionSegment]] = [ObjectDetectionSegment.from_dataset(x) for x in dataset.get_limited_dataset()]

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

    # Save metrics to a DataFrame
    metrics_df = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }, index=[0])
    
    return metrics_df

def get_run_info(start_time, monitor: SystemMonitor):
    run_time = time.perf_counter() - start_time
    system_info = platform.uname()
    cpu_avg, mem_avg = monitor.get_avg_usage()

    run_info_df = pd.DataFrame({
        'run_time': run_time,
        'cpu_usage': cpu_avg,
        'mem_usage': mem_avg,
        'system_info': str(system_info),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, index=[0])

    return run_info_df

for name, model in models.items():
    model: ObjectRecognition
    start = time.perf_counter()
    
    images = [i['image'] for i in dataset.get_limited_dataset()]
    print(f"Processing {len(images)} images with {name}")
    
    monitor = SystemMonitor()
    monitor.start()

    results = model.run_model(images)
    print(f"Time taken: {time.perf_counter() - start:.3f}s")
    monitor.stop()

    # Calculate metrics
    metrics_df = calculate_metrics(results, ground_truths)

    # Get run info
    run_info_df = get_run_info(start, monitor)

    # Combine metrics and run info
    total_results_df = pd.concat([run_info_df, metrics_df], axis=1)

    # Create a new ExcelWriter object
    writer = pd.ExcelWriter(EVAL_FOLDER/f'{name}_total_results.xlsx', engine='openpyxl')  # type: ignore

    # Write total results to the first sheet
    total_results_df.to_excel(writer, sheet_name='Total_Results', index=False)

    # Write metrics to the second sheet
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

    # Create a DataFrame for all image results
    all_image_results_df = pd.DataFrame()

    for idx, image_results in enumerate(results):
        image_df = pd.DataFrame([result.model_dump() for result in image_results])
        image_df['image_index'] = idx
        all_image_results_df = pd.concat([all_image_results_df, image_df], ignore_index=True)

    # Write all image results to the third sheet
    all_image_results_df.to_excel(writer, sheet_name='All_Image_Results', index=False)

    # Save and close the writer
    writer.close()

    print(f"Precision: {metrics_df['precision'].values[0]:.3f}, Recall: {metrics_df['recall'].values[0]:.3f}, F1 Score: {metrics_df['f1_score'].values[0]:.3f}")
