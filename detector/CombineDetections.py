from typing import Union, List
import torch
import numpy as np
from detector.PatchDetector import PatchDetector


class CombineDetections:


    def __init__(
        self,
        element_crops: Union[PatchDetector, List[PatchDetector]],
        nms_threshold=0.3,
        match_metric='IOS',
        intelligent_sorter=True,
        sorter_bins=5,
        class_agnostic_nms=True
    ) -> None:

        self.nms_threshold = nms_threshold  # IOU or IOS treshold for NMS
        self.match_metric = match_metric 
        self.intelligent_sorter = intelligent_sorter # enable sorting by area and confidence parameter
        self.sorter_bins = sorter_bins
        self.class_agnostic_nms = class_agnostic_nms

        # Check if element_crops is a list
        if isinstance(element_crops, list):
            # Ensure all elements in the list have the same source_image and other params
            first_image = element_crops[0].crops[0].source_image
            for element in element_crops:
                if not np.array_equal(element.crops[0].source_image, first_image):
                    raise ValueError(
                        "The source images in element_crops differ, "
                        "so combining results from these objects is not possible."
                    )
                if not element.resize_initial_size:
                    raise ValueError(
                        "When working with a list of element_crops, "
                        "resize_initial_size should be True everywhere."
                    )
            
            self.class_names = element_crops[0].class_names_dict
            self.crops = [crop for element in element_crops for crop in element.crops]
            self.image = element_crops[0].crops[0].source_image
        else:
            self.class_names = element_crops.class_names_dict
            self.crops = element_crops.crops  # List to store the CropElement objects
            self.image = element_crops.crops[0].source_image

        # Combinate detections of all patches
        (
            self.detected_conf_list_full,
            self.detected_xyxy_list_full,
            self.detected_masks_list_full,
            self.detected_cls_id_list_full,
            self.detected_polygons_list_full
        ) = self.combinate_detections(crops=self.crops)

        self.detected_cls_names_list_full = [
            self.class_names[value] for value in self.detected_cls_id_list_full
        ]  # make str list

        # Invoke the NMS:
        if self.class_agnostic_nms:
            self.filtered_indices = self.nms(
                torch.tensor(self.detected_conf_list_full),
                torch.tensor(self.detected_xyxy_list_full),
                self.match_metric,
                self.nms_threshold,
                self.detected_masks_list_full,
                intelligent_sorter=self.intelligent_sorter
            ) 

        else:
            self.filtered_indices = self.not_agnostic_nms(
                torch.tensor(self.detected_cls_id_list_full),
                torch.tensor(self.detected_conf_list_full),
                torch.tensor(self.detected_xyxy_list_full),
                self.match_metric,
                self.nms_threshold,
                self.detected_masks_list_full,
                intelligent_sorter=self.intelligent_sorter
            )  

        # Apply filtering (nms output indeces) to the prediction lists
        self.filtered_confidences = [self.detected_conf_list_full[i] for i in self.filtered_indices]
        self.filtered_boxes = [self.detected_xyxy_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_id = [self.detected_cls_id_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_names = [self.detected_cls_names_list_full[i] for i in self.filtered_indices]

        # Masks filtering:
        self.filtered_masks = []

        # Polygons filtering:
        self.filtered_polygons = []

    def combinate_detections(self, crops):

        detected_conf = []
        detected_xyxy = []
        detected_masks = []
        detected_cls = []
        detected_polygons = []

        for crop in crops:
            detected_conf.extend(crop.detected_conf)
            detected_xyxy.extend(crop.detected_xyxy_real)
            detected_masks.extend(crop.detected_masks_real)
            detected_cls.extend(crop.detected_cls)
            detected_polygons.extend(crop.detected_polygons_real)

        return detected_conf, detected_xyxy, detected_masks, detected_cls, detected_polygons

    @staticmethod
    def average_to_bound(confidences, N=10):

        # Create the bounds
        step = 1 / N
        bounds = np.arange(0, 1 + step, step)

        # Use np.digitize to determine the corresponding bin for each value
        indices = np.digitize(confidences, bounds, right=True) - 1

        # Bind values to the left boundary of the corresponding bin
        averaged_confidences = np.round(bounds[indices], 2) 

        return averaged_confidences.tolist()

    @staticmethod
    def intersect_over_union(mask, masks_list):

        iou_scores = []
        for other_mask in masks_list:
            # Compute intersection and union
            intersection = np.logical_and(mask, other_mask).sum()
            union = np.logical_or(mask, other_mask).sum()
            # Compute IoU score, avoiding division by zero
            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)
        return torch.tensor(iou_scores)

    @staticmethod
    def intersect_over_smaller(mask, masks_list):

        ios_scores = []
        for other_mask in masks_list:
            # Compute intersection and area of smaller mask
            intersection = np.logical_and(mask, other_mask).sum()
            smaller_area = min(mask.sum(), other_mask.sum())
            # Compute IoU score over smaller area, avoiding division by zero
            ios = intersection / smaller_area if smaller_area != 0 else 0
            ios_scores.append(ios)
        return torch.tensor(ios_scores)

    def nms(
        self,
        confidences: torch.tensor,
        boxes: torch.tensor,
        match_metric,
        nms_threshold,
        masks=[],
        intelligent_sorter=False, 
        cls_indexes=None 
    ):


        if len(boxes) == 0:
            return []

        # Extract coordinates for every prediction box present
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate area of every box
        areas = (x2 - x1) * (y2 - y1)

        # Sort the prediction boxes according to their confidence scores or intelligent_sorter mode
        if intelligent_sorter:
            # Sort the prediction boxes according to their round confidence scores and area sizes
            order = torch.tensor(
                sorted(
                    range(len(confidences)),
                    key=lambda k: (
                        self.average_to_bound(confidences[k].item(), self.sorter_bins),
                        areas[k],
                    ),
                    reverse=False,
                )
            )
        else:
            order = confidences.argsort()
        # Initialise an empty list for filtered prediction boxes
        keep = []

        while len(order) > 0:
            # Extract the index of the prediction with highest score
            idx = order[-1]

            # Push the index in filtered predictions list
            keep.append(idx.tolist())

            # Remove the index from the list
            order = order[:-1]

            # If there are no more boxes, break
            if len(order) == 0:
                break

            # Select coordinates of BBoxes according to the indices
            xx1 = torch.index_select(x1, dim=0, index=order)
            xx2 = torch.index_select(x2, dim=0, index=order)
            yy1 = torch.index_select(y1, dim=0, index=order)
            yy2 = torch.index_select(y2, dim=0, index=order)

            # Find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])

            # Find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1

            # Take max with 0.0 to avoid negative width and height
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            # Find the intersection area
            inter = w * h

            # Find the areas of BBoxes
            rem_areas = torch.index_select(areas, dim=0, index=order)

            if match_metric == "IOU":
                # Find the union of every prediction with the prediction
                union = (rem_areas - inter) + areas[idx]
                # Find the IoU of every prediction
                match_metric_value = inter / union

            elif match_metric == "IOS":
                # Find the smaller area of every prediction with the prediction
                smaller = torch.min(rem_areas, areas[idx])
                # Find the IoU of every prediction
                match_metric_value = inter / smaller

            else:
                raise ValueError("Unknown matching metric")

            # If masks are provided and IoU based on bounding boxes is greater than 0,
            # calculate IoU for masks and keep the ones with IoU < nms_threshold
            if len(masks) > 0 and torch.any(match_metric_value > 0):

                mask_mask = match_metric_value > 0 

                order_2 = order[mask_mask]
                filtered_masks = [masks[i] for i in order_2]

                if match_metric == "IOU":
                    mask_iou = self.intersect_over_union(masks[idx], filtered_masks)
                    mask_mask = mask_iou > nms_threshold

                elif match_metric == "IOS":
                    mask_ios = self.intersect_over_smaller(masks[idx], filtered_masks)
                    mask_mask = mask_ios > nms_threshold
                # create a tensor of indences to delete in tensor order
                order_2 = order_2[mask_mask]
                inverse_mask = ~torch.isin(order, order_2)

                # Keep only those order values that are not contained in order_2
                order = order[inverse_mask]

            else:
                # Keep the boxes with IoU/IoS less than threshold
                mask = match_metric_value < nms_threshold

                order = order[mask]
        if cls_indexes is not None:
            keep = [cls_indexes[i] for i in keep]
        return keep

    def not_agnostic_nms(
            self,
            detected_cls_id_list_full,
            detected_conf_list_full, 
            detected_xyxy_list_full, 
            match_metric, 
            nms_threshold, 
            detected_masks_list_full, 
            intelligent_sorter
                     ):

        all_keeps = []
        for cls in torch.unique(detected_cls_id_list_full):
            cls_indexes = torch.where(detected_cls_id_list_full==cls)[0]
            if len(detected_masks_list_full) > 0:
                masks_of_class = [detected_masks_list_full[i] for i in cls_indexes]
            else:
                masks_of_class = []
            keep_indexes = self.nms(
                    detected_conf_list_full[cls_indexes],
                    detected_xyxy_list_full[cls_indexes],
                    match_metric,
                    nms_threshold,
                    masks_of_class,
                    intelligent_sorter,
                    cls_indexes
                )
            all_keeps.extend(keep_indexes)
        return all_keeps
    
    def __str__(self):
        # Print the list of useful attributes (non-empty ones)
        useful_attributes = []
        if self.filtered_confidences:
            useful_attributes.append("filtered_confidences")
        if self.filtered_boxes:
            useful_attributes.append("filtered_boxes")
        if self.filtered_classes_id:
            useful_attributes.append("filtered_classes_id")
        if self.filtered_classes_names:
            useful_attributes.append("filtered_classes_names")
        if self.filtered_masks:
            useful_attributes.append("filtered_masks")
        if self.filtered_polygons:
            useful_attributes.append("filtered_polygons")

        # If all attributes are empty
        if not useful_attributes:
            return "Useful attributes: nothing detected in the frame"

        # Build the output string
        output = "Useful attributes: " + ", ".join(useful_attributes) + "\n\n"
        for attr in useful_attributes:
            value = getattr(self, attr)
            if attr == "filtered_masks":
                output += f"{attr}: the list of binary masks with shape {value[0].shape} (length: {len(value)})\n"
            elif len(value) > 10:
                list_text = f"{value[:10]}"
                output += f"{attr}: {list_text[:-1]}, ...] (length: {len(value)})\n"
            else:
                output += f"{attr}: {value} (length: {len(value)})\n"
        return output
