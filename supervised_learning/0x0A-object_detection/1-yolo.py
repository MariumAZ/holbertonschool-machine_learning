#!/usr/bin/env python3


""" contains Yolo class"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """Class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            r = f.read().split("\n")[:-1]
        self.class_names = r
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """sigmoid"""
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """it processes output"""
        inp_w = self.model.input.shape[1].value
        inp_h = self.model.input.shape[2].value
        boxes = []
        box_c = []
        box_c_p = []
        for output in outputs:
            boxes.append(output[..., 0:4])
            box_c.append(self.sigmoid(output[..., 4:5]))
            box_c_p.append(self.sigmoid(output[..., 5:]))
        img_w = image_size[1]
        img_h = image_size[0]
        for i in range(len(outputs)):
            grid_h = boxes[i].shape[0]
            grid_w = boxes[i].shape[1]
            a = boxes[i].shape[2]
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            tx = boxes[i][..., 0]
            ty = boxes[i][..., 1]
            tw = boxes[i][..., 2]
            th = boxes[i][..., 3]
            cx = np.indices((grid_h, grid_w, a))[1]
            cy = np.indices((grid_h, grid_w, a))[0]
            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h
            input_w = self.model.input.shape[1].value
            input_h = self.model.input.shape[2].value
            bw = anchor_w * np.exp(tw) / input_w
            bh = anchor_h * np.exp(th) / input_h
            x1 = bx - bw / 2
            x2 = x1 + bw
            y1 = by - bh / 2
            y2 = y1 + bh
            boxes[i][..., 0] = x1 * img_w
            boxes[i][..., 1] = y1 * img_h
            boxes[i][..., 2] = x2 * img_w
            boxes[i][..., 3] = y2 * img_h
        return boxes, box_c, box_c_p
