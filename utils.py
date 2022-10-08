import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    # sort bounding boxes based on confidence score
    pairs = [x for x in list(zip(confidence_score, bounding_boxes)) if x[0] > threshold]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    boxes, scores = [], []
    visited = set()
    # loop to visit all bounding boxes
    while len(visited) < len(pairs):
        idx = 0
        # get the first index that is not visited before
        for i in range(len(pairs)):
            if i in visited:
                continue
            idx = i
            break
        # add this bounding box to output list
        boxes.append(pairs[idx][1])
        scores.append(pairs[idx][0])
        visited.add(idx)
        # loop over other bounding boxes to remove overlapping bboxes
        for i in range(len(pairs)):
            if i in visited:
                continue
            # ignore overlapping bboxes with iou > 0.3
            if iou(pairs[idx][1], pairs[i][1]) > 0.3:
                visited.add(i)

    return boxes, scores


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    # get the boarder coordiantes of union of two bboxes
    x_left = max(box1[0], box2[0]) 
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # area is zero if two bboxes cannot be unioned
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # calculate intersection area
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # iou is intersection divided by union (addition of areas of two bboxes minus intersection)
    iou = intersection / (area1 + area2 - intersection)
    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id": classes[i],
        } for i in range(len(classes))
        ]

    return box_list
