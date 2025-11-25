import torch
import time
import cv2
import numpy as np
def intersection_over_union(boxes_preds, boxes_labels, box_formats="corners"):
    if box_formats == "midpoint":
        box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3]/2
        box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4]/2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[...,2:3]/2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[...,3:4]/2
        box2_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3]/2
        box2_y1 = boxes_labels[...,1:2] - boxes_labels[...,3:4]/2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[...,2:3]/2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[...,3:4]/2
    if box_formats == "corners":
        box1_x1 = boxes_preds[...,0:1]
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[...,3:4]
        box2_x1 = boxes_labels[...,0:1]
        box2_y1 = boxes_labels[...,1:2]
        box2_x2 = boxes_labels[...,2:3]
        box2_y2 = boxes_labels[...,3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = max(x2 - x1,0) * max(y2 - y1, 0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def nms(bboxes, iou_threshold =0.1, prob_threshold=0.6, box_formats = "midpoint"):
    bboxes = [box for box in bboxes if box[4] > prob_threshold]
    new_bboxes = []
    bboxes_after_nms = []
    # convert (xyxy + conf + 4 class) => (xyxy+conf+(class dung))
    for box in bboxes:
        new_box = []
        new_box[:5] = box[:5]
        tmp = max(box[5:])
        new_box.append(np.argmax(box[5:]))
        new_bboxes.append(new_box)

    new_bboxes = sorted(new_bboxes, key= lambda x:x[4], reverse = True)
    while new_bboxes:
        chosen_box = new_bboxes.pop(0)
        new_bboxes = [
            box
            for box in new_bboxes
            if box[5] != chosen_box[5] 
            or intersection_over_union(
            torch.tensor(box[:4]),
            torch.tensor(chosen_box[:4]),
             box_formats=box_formats,
               ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def perspective_img(img, keypoints):
    """
    @param img: input image
    @param keypoints: list of 4 points
    """

    # if isinstance(keypoints, list):
    #     keypoints = np.array(keypoints)
    # rect = reorder_points(keypoints)
    (tl, tr, br, bl) = keypoints

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order

    dst_points = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1] ], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(keypoints, dst_points)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

