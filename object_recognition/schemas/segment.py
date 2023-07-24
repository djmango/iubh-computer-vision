from pydantic import BaseModel, Field

class ObjectDetectionSegment(BaseModel):
    """ Schema for object detection segment, includes:
        label (Human readable class label)
        bounding_box (xmin, ymin, xmax, ymax)
        confidence (Model confidence in detection)
        class_id (Specific to model)
        model (Reference to model used)
    """
    label: str = Field(..., description="Class name of object")
    bounding_box: list = Field(..., description="Bounding box of object")
    confidence: float = Field(..., description="Confidence of object")
    class_id: int = Field(..., description="Class id of object")
