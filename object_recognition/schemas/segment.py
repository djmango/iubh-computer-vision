from pydantic import BaseModel, Field

from object_recognition.dataset import categories

class ObjectDetectionSegment(BaseModel):
    """ Schema for object detection segment, includes:
        label (Human readable class label)
        bounding_box (xmin, ymin, xmax, ymax)
        confidence (Model confidence in detection)
    """
    label: str = Field(..., description="Class name of object")
    bounding_box: list = Field(..., description="Bounding box of object")
    confidence: float = Field(..., description="Confidence of object")

    @classmethod
    def from_dataset(cls, dataset):
        """
        From the HuggingFace Dataset format to our pydantic format for ease of use and simplicity
        {
            'height': 663,
             'image': <PIL image>
             'image_id': 15,
             'objects': {
                  'area': [3796, 1596, 152768, 81002],
                  'bbox': [[302.0, 109.0, 73.0, 52.0],
                   [810.0, 100.0, 57.0, 28.0],
                   [160.0, 31.0, 248.0, 616.0],
                   [741.0, 68.0, 202.0, 401.0]],
                  'category': [4, 4, 0, 0],
                  'id': [114, 115, 116, 117]},
             'width': 943
         }
         """
         
        image_segments: list[cls] = []
        for obj in range(len(dataset['objects']['bbox'])):
            label = categories.int2str(dataset['objects']['category'][obj]) # assuming you have the categories object
            bounding_box = dataset['objects']['bbox'][obj] # Coco bounding box format is [xmin, ymin, width, height]
            bounding_box[2] += bounding_box[0] # Convert width to xmax
            bounding_box[3] += bounding_box[1] # Convert height to ymax
            confidence = 0 # There is no confidence in the dataset
            segment = cls(label=label, bounding_box=bounding_box, confidence=confidence)
            image_segments.append(segment)

        return image_segments
