import numpy as np
import cv2
from utils.elapsed import TimeElapsed


class SliceElement:
    # Class containing information about a specific crop
    def __init__(
        self,
        source_image: np.ndarray,
        crop: np.ndarray,
        x_start: int,
        y_start: int,
    ) -> None:
        self.source_image = source_image  # Original image 
        self.crop = crop  # Specific crop 
        self.x_start = x_start  # Coordinate of the top-left corner X
        self.y_start = y_start  # Coordinate of the top-left corner Y

        # YOLO output results:
        self.detected_conf = None  # List of confidence scores of detected objects
        self.detected_cls = None  # List of classes of detected objects
        self.detected_xyxy = None  # List of lists containing xyxy box coordinates
        self.detected_masks = None # List of np arrays containing masks in case of yolo-seg
        self.polygons = None # List of polygons points in case of using memory optimaze
        
        # Refined coordinates according to crop position information
        self.detected_xyxy_real = None  # List of lists containing xyxy box coordinates in values from source_image_resized or source_image
        self.detected_masks_real = None # List of np arrays containing masks in case of yolo-seg with the size of source_image_resized or source_image
        self.detected_polygons_real = None # List of polygons points in case of using memory optimaze in values from source_image_resized or source_image


    def calculate_real_values(self):
        # Вычисляет реальные значения ограничивающих прямоугольников на входном изображении
        x_start_global = self.x_start  # Global X coordinate of the crop
        y_start_global = self.y_start  # Global Y coordinate of the crop

        self.detected_xyxy_real = []  # List of lists with xyxy box coordinates in the values ​​of the source_image_resized
        self.detected_masks_real = []  # List of np arrays with masks in case of yolo-seg sized as source_image_resized
        self.detected_polygons_real = [] # List of polygons in case of yolo-seg sized as source_image_resized

        for bbox in self.detected_xyxy:
            # Calculate real box coordinates based on the position information of the crop
            x_min, y_min, x_max, y_max = bbox
            x_min_real = x_min + x_start_global
            y_min_real = y_min + y_start_global
            x_max_real = x_max + x_start_global
            y_max_real = y_max + y_start_global
            self.detected_xyxy_real.append([x_min_real, y_min_real, x_max_real, y_max_real])

        if self.polygons is not None:
            # Adjust the mask coordinates
            for mask in self.polygons:
                mask[:, 0] += x_start_global  # Add x_start_global to all x coordinates
                mask[:, 1] += y_start_global  # Add y_start_global to all y coordinates
                self.detected_polygons_real.append(mask.astype(np.uint16))
        