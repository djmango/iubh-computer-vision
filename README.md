# Project: Computer Vision (DLBAIPCV)

IUBH Computer Vision project, evalulate cutting edge of open-source computer vision models.
Sulaiman Ghori July 2023

## Video Demo
[![Video Demo](https://img.youtube.com/vi/GEDbN8ePU4s/0.jpg)](https://youtu.be/GEDbN8ePU4s)


### Introduction

Object recognition is a critical task in various computer vision (CV) applications, including autonomous driving, mass surveillance networks, and medical imaging. Object recognition in CV is the ability to detect, segment, and identify entities in video sequences. It’s useful for creating intelligent systems that can interact with and understand their environment. In this project, I aim to create a computer vision system that can take video sequences as an input and then output them augmented with information on the position, shape, and name of the entities present in the video.

The goal of this work is to evaluate three state-of-the-art (SoA) CV algorithms for object detection, using a dataset representative of a wide range of use cases, and paired with ground truth information, in order to conduct the quantitative measurements of the aforementioned evaluation. High-level, this evaluation will focus on the validity, reliability, and objectivity of the selected algorithms, and I will later define these terms specifically for this context. My qualitative analysis of these SoA approaches will be presented to compare and contrast their strengths and weaknesses, and again, paired with a set of quantitative tests, which I will describe in detail.

I have chosen to evaluate existing open-source SoA approaches because they represent the cutting edge in object recognition, and have already proven their merit in various applications. By studying and comparing these approaches, I aim to identify the best-suited algorithm for my computer vision system, which will enable me to achieve superior performance in object recognition tasks. I mention open-source, because while a valid point could be made that superior systems have been or are being actively developed by the expert teams at leading companies such as Meta, Tesla, and Google, it has been, and continues to be the case that these teams typically share their research and development findings with the open-source community in the form of academic publishings and open-sourcing of their architecture. Indeed, many of the developments pioneered by these teams have been key inflection points for the development of intelligent computer systems, such as Attention is All You Need. The rising tide raises all ships.

To accomplish my objectives, I lay out the following steps:

1. I will first lay out the specific problem, in terms of high-level goal, objective of the CV system, evaluation methods, and success criteria.
2. I will then conduct a thorough survey of open-source CV solutions, and evaluating based on performance in generalized entity detection and segmentation, selecting the best three for the next step.
3. Once the CV systems have been chosen, I will select an industry-vetted CV dataset representing a wide range of real-world situations, and use the included ground truth information to conduct the evaluations
4. I will then engage in a software engineering project to build an evaluation framework. This will load the dataset, conduct preprocessing on the selected image set, run the models in an efficient manner, and finally output the evaluations in a standard format.
5. To wrap up the analysis section of the project, I will perform a qualitative review of the CV approaches, discussing their strengths and weaknesses based on my own observations, and surfacing insights from the research I studied during the survey step.
6. Finally, after reviewing the quantitative data, I will select the best performing model and program a real-time object-detection in video demo to showcase its capabilities in a number of real-world scenarios, presented in the format of a project report video.

With the conclusion of this project, I shall present a robust and generalized CV system that can accurately recognize entities in video sequences in real-time, thus contributing to the advancement of CV in regards to its practical application.

### Problem Definition and Objective

Allow me to first illustrate the high-level goal. In the vast domain computer vision, today one particular task holds our attention, that being object recognition. Visual perception, a seemingly effortless task performed by the brain, has been the subject of extensive research and exploration by computer scientists worldwide. Looking back upon the last decade, a significant portion of this research was dedicated to the development of vision systems that can skillfully ingest a video sequence and return with the position, shape, and moniker of the objects present. Applications of such are clear: intelligent, autonomous agents. The skilled teams at Tesla and Comma use such for their autonomous driving programs, Boston Dynamics for their multi-legged task-completing robots, and a wide variety of others for an equally colorful assortment of tasks. This, dear reader, is our high-level goal: a robust system that provides us with real-time, accurate object recognition.

Our system should be capable of, one, detecting the presence of an object in a video sequence, two, demarcating the contours of the object, this commonly being referred to as a bounding box in two-dimension vision problems, and three, attributing it the correct label, that being a human-understandable name for the object. This is no simple feat. Consider the variety and variability of objects in the real world. Consider the challenges posed by lighting conditions, camera angles, and occlusions. Yet, it is precisely such a demanding set of requirements that makes this objective all the more appealing.

For each detected object, I will assess the overlap between the predicted label and the true label. Precision, recall, and F1 score will be our metrics of choice for the quantitative assessment, their reliability and interpretability rendering them ideal for this purpose. Precision is a measure of how many of our predicted labels are indeed correct, true positives over true positives plus false negatives. Recall quantifies the fraction of true labels that the system managed to predict correctly, true positives over true positives plus false negatives. The F1 score, a harmonic mean of precision and recall, gives us a single number that balances these two aspects, a solid comprehensive score we can use to compare overall performance.

To take our evaluation a step further, one might consider an analysis based on physical box overlap. In this approach, I would compare the bounding boxes predicted by our model with the actual boundaries of the objects, taking into account not only the accuracy of the labels but also the precision of the spatial localization. In the present form of this project, I do not aim to conduct this evaluation, however it remains an opportunity if one desires to expand the scope.

In terms of success criteria, I aim to match or exceed industry standards for each of these metrics. Top-performing models on the COCO dataset, such as YOLOv5, achieved average precision scores above 60%. Precision and recall values in the high 80s or 90s were not uncommon. Therefore, I will aim for similar performance from our system.

### Model Survey

Now that the problem statement and success criteria has been defined, the next logical step was to examine the landscape of open-source computer vision solutions to determine which might best serve our needs.

After a comprehensive review, three models emerged as the most promising: YOLOS (tiny-sized model), DETR (End-to-End Object Detection model with ResNet-50 backbone), and Mask2Former (COCO-trained panoptic segmentation).

The YOLOS model is a lightweight yet powerful option. Fine-tuned on the COCO object detection dataset, which consists of 118k annotated images, YOLOS uses the DETR loss to train a Vision Transformer (ViT). It accomplishes this with a simple but effective approach, employing a "bipartite matching loss" to compare the predicted classes and bounding boxes of each object query to the ground truth annotations.

Moving onto DETR, it stands out as an end-to-end object detection solution with a ResNet-50 backbone. Trained on the same COCO object detection dataset as YOLOS, DETR employs an encoder-decoder transformer with a convolutional backbone to achieve excellent object detection.

Finally, we have the Mask2Former model. Unique in its approach, Mask2Former treats instance, semantic, and panoptic segmentation tasks as if they were all instance segmentation. It uses a novel technique that involves predicting a set of masks and corresponding labels to achieve this. I found this mask-based architecture particularly fascinating, and interesting that it performed so well. It too, was trained on the COCO dataset.

These three models, chosen for their outstanding performance in entity detection and segmentation, are by no means the only options available in the rich landscape of open-source CV solutions. However, they represent some of the most advanced and efficient tools currently accessible. Their strong performance in critical areas of our project makes them a viable option for this endeavor.

### Dataset Selection: Common Objects in Context (COCO)

In order to evaluate the performance of our chosen CV systems, we need a robust and comprehensive dataset. After survey, the selected dataset for this task is the industry-vetted Common Objects in Context (COCO), a large-scale object detection, segmentation, and captioning dataset known for its breadth and diversity. It was used as the training set for all of the selected models, as well as many more, making it a perfect fit.

The COCO dataset stands out because it emulates the complexity of the real world in its data. Instead of offering images with single, isolated objects, COCO provides scenarios where multiple objects interact in a wide range of contexts. In addition, COCO is incredibly extensive, comprising over 330,000 images, 1.5 million object instances spanning 80 object categories. This broad scope promises a varied and comprehensive representation of real-world situations, contributing to a more rigorous evaluation of our selected CV systems. Uniquely, COCO encompasses 91 additional auxiliary categories, which refers to images of non-object items such as grass, sky, or walls. This supplementary dimension offers an extra layer of complexity, further improving the robustness of our system evaluation.

The use of the COCO dataset for system evaluation validates that our models are well-equipped to handle a wide array of real-world situations.

### Building the Evaluation Framework

The creation of a sophisticated evaluation framework was a crucial component of this project. The complex challenges I faced during this stage required a blend of software engineering skills, novel programming techniques, and meticulous attention to detail.

The project was structured as Python 3.11 package, comprising of multiple interconnected modules, each serving a distinct purpose within the overall system. These modules are organized in a hierarchical fashion, resembling a well-structured tree. The root directory contains a 'README.md' file detailing the project and how to use it, as well as a 'pyproject.toml' and 'poetry.lock' file which handle the project dependencies and environment, often a complexity and frequent annoyance in machine learning projects. Poetry aids this significantly.

The 'eval' directory, one of the main branches of our project tree, contains the results of the evaluations. Each run generates a new subdirectory, named with the corresponding timestamp of the run. Inside these subdirectories, we can find 'xlsx' files storing the detailed results from each CV model. These result files are specific to a timestamped run and a model. Each contains a sheet with overall run metrics, such as compute and time information as well as the evaluation metrics, precision recall and F1 score. Another sheet stores the inferences of each image in the selected set, which can be used for additional later evaluation.

To streamline our coding process and foster reusability, I followed best software development practices. I implemented design patterns like Singletons and Abstract classes, ensuring an efficient and flexible codebase. The separation of responsibilities principle was also rigorously applied, so each class and function had a clear, defined purpose.

The 'object_recognition' package I developed as the backbone of our project, hosting several Python files and modules, each assigned to handle a particular task. For instance, 'dataset.py' is responsible for managing the dataset operations, 'device.py' facilitates the interaction with the hardware, 'eval.py' conducts the evaluations, and 'video.py' processes video input if required. The 'models' subdirectory within this houses the individual CV models' implementations. For the CV models, I first constructed an abstract class outlining all the expected functionality, IO structure, and inference output as a Pydantic model. I then implemented each model using that abstract class as a base. It is extensible to additional models, allowing for reuse in future endeavors.

Git was employed for version control, ensuring smooth management and tracking of code changes during the project's lifecycle. The project is hosted on GitHub, at djmango/iubh-computer-vision. I utilized the open-source 'transformers' and 'datasets' libraries by HuggingFace to handle the CV models and data, and 'fuzzywuzzy' for fuzzy matching the labels of the outputs via Levenshtein distance. To tackle the challenge of memory optimization, I applied batching and invoked garbage collection. For accelerated inference, I took advantage of the powerful features of PyTorch CUDA and MPS.

Given the enormous size of the COCO dataset, memory management presented a significant hurdle. To address this, I developed custom streamers and batching systems to handle data in manageable chunks, reducing the memory footprint. Moreover, several memory optimization methods were introduced to prevent memory overflows and optimize resource allocation. Despite this, I had to limit the size of the evaluation to under 1000 images at one time in order to keep memory usage below 64Gb. Additionally, I encountered segmentation faults when running with CUDA, which required substantial debugging and system knowledge to resolve. These challenges, though tough, provided an enriching learning experience and led to a more robust final product.

Looking forward, the recent announcement of Mojo, a new programming language from Chris Lattner's company, Modular, excites me. It is a Python superset, and promises the speed of C with the simplicity of Python. It is specifically designed for highly accelerated machine learning computation. Such advancements point towards the intriguing possibility of more efficient and streamlined CV model evaluation in future projects. I’ve recently reviewed interviews with Chris Lattner, who previously developed the software for Google’s TPU accelerators, which is another exciting development I’ve had the pleasure to work with. The principles of simplicity and modularity are solid.

### **Quantitative Analysis**

In this section I’d like to briefly review the quantitative results of the three models I ran with the evaluations system described above on a set of 100 random COCOs validation images.

Mask2Former had a precision of 0.627. This relatively lower precision indicates a higher rate of false positives, as seen with the 41 recorded for this model. Despite this, it had a high recall of 0.863, meaning it was able to detect a large number of objects in the images. Its F1 score was 0.726, and it accounted for 69 true positives and 11 false negatives.

DETR Resnet displayed a high precision of 0.966, suggesting that it rarely mislabels the objects it detects. The recall, stood at 0.877, indicating that while its predictions were accurate, it still did miss a number of objects, though noteworthy that it is the best recall of the three. This contributed to an F1 score of 0.919, showcasing the balance between precision and recall. The model identified 57 true positives, and registered 2 false positives and missed 8 objects, marking them as false negatives.

Yolos exhibited exceptional precision at 0.978, suggesting that it very rarely mislabels the objects it identifies. Though, its recall was 0.759, pointing towards a more modest ability to detect all objects within the image when compared to the other two models. The balance between these two led to an F1 score of 0.854. The model reported 44 true positives, only 1 false positive, and 14 false negatives.

### **Qualitative Analysis**

Concluding the analytical component of the project, I embark on a qualitative assessment of the selected computer vision models – YOLOs, DETR-ResNet, and Mask2Former. This review centers on their performance characteristics, the nuances of their design, and the overall experience of their integration and operation. It also encapsulates the insights I gained from the research conducted during the survey phase of the project.

YOLOs stood out in terms of speed, outpacing both DETR-ResNet and Mask2Former. This factor can be particularly critical in applications where real-time or near-real-time processing is essential, for example the video demo, as it allows for swift responses based on the models' outputs. In terms of integration and use, both YOLOs and DETR-ResNet were relatively straightforward and user-friendly, making them viable choices for projects with tight timelines or limited resources.

Mask2Former, though having a slower processing speed, offered an interesting twist with its panoptic segmentation architecture. This unique aspect introduced an additional layer of complexity, especially when it came to parsing the results. However, this could potentially be offset by the rich, nuanced outputs it could produce due to it’s end-to-end utilization of masks, which may provide more insights in certain application scenarios.

Notably, Mask2Former was trained on a slightly different cut of the COCO dataset than YOLOs and DETR-ResNet. To account for this and ensure accurate evaluations, it was necessary to apply fuzzy matching for labels. Despite the extra work this necessitated, it served as an exercise in dealing with data inconsistencies and the need to align different data sources.

In terms of output accuracy, Mask2Former showed a high recall rate, suggesting that it is proficient at identifying relevant objects within an image. However, it exhibited a low precision rate, indicating a tendency to over-identify, or be "over-eager," thereby potentially increasing the noise within the results. Both YOLOs and DETR-ResNet presented a more balanced performance in this respect, showcasing their robustness and reliability.

Ultimately, YOLOs emerged as the top performer among the models in this project. Its combination of rapid processing speed, ease of use, and robust performance metrics make it an attractive choice for a variety of computer vision tasks.

In conclusion, this qualitative review underscores the diversity of the field of computer vision and the range of capabilities that different models offer. Each has its strengths and weaknesses, and the choice of model will depend on the requirements and constraints of the task at hand.

### Real-Time Object-Detection Video Demonstration

For the final stage, I created a real-time object-detection video demonstration, reusing some of the software developed for the evaluation framework and writing a new video module using OpenCV (cv2). To power the demo, the YOLOs model was selected based on its fast processing, ease-of-use, and favorable performance metrics.

The development of the video demo was relatively smooth. The primary challenge lies in optimizing the implementation for computational efficiency. I measured the demo as running at reliably at 15 frames per second on my M1 Mac. It's worth mentioning that there doesn’t yet exist a YOLOs implementation specifically optimized for the M1 Mac's ARM architecture. During the evaluation stage, I experimented with recompiling the models for M1 and utilizing MPS, resulting in a 2-4x speedup in inference. There were significant challenges with memory, and thus I moved away from this approach. Additionally, the current codebase is not designed to handle live streaming data and inferences. Keeping these areas for improvement in mind, the system's current state is a viable starting point for further development.

### **Conclusion**

The conclusion of this project leaves numerous paths for further exploration, from optimizing the YOLOs model for specific hardware architectures, to developing more sophisticated object recognition and tracking systems, and expanding the evaluation framework.

In the continuously evolving field of computer vision, this project serves as an excellent reminder to me about the essence of the field – the combination of rigorous analytical investigation and creative problem-solving to develop technology that can truly see and understand the world like we do oh so effortlessly.
