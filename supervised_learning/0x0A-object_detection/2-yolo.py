#!/usr/bin/env python3
import numpy as np
import tensorflow.keras as K 

class Yolo():
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path : path to the Darknet keras model 

        """
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.anchors = anchors
        self.nms_t = nms_t
        with open(classes_path,'r') as c:
            self.class_names = [i.strip() for i in c]
    def sigmoid(self,x):
        return 1 / ( 1 + np.exp(-x))        
    def process_outputs(self, outputs, image_size):
        #initialize empty lists:
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i,output in enumerate(outputs):

            boxes.append(output[...,0:4]) #extract 4 coordinates (x,y,h,w)
            box_confidences.append(self.sigmoid(output[...,4:5])) #extract probability
            box_class_probs.append(self.sigmoid(output[...,5:]))  #extract prob classes 
            grid_h = output.shape[0]   #grid height
            grid_w = output.shape[1]   #grid weight
            anchor_boxes = output.shape[2] #number of anchors on each cell
            #network prediction outputs
            t_x = output[...,0] 
            t_y = output[...,1]
            t_w = output[...,2]
            t_h = output[...,3]
            
            cx = np.indices((grid_h, grid_w, anchor_boxes))[1]
            cy = np.indices((grid_h, grid_w, anchor_boxes))[0]
            
            #bounding box coordinates(x,y):
            bx = self.sigmoid(t_x) + cx
            by = self.sigmoid(t_y) + cy 
            #extract anchors dimensions:
            a_w = self.anchors[i,:,0]
            a_h = self.anchors[i,:,1]
            bw = a_w * np.exp(t_w) 
            bh = a_h * np.exp(t_h) 
           

            # Normalizing
            bx = bx / grid_w
            by = by / grid_h
            bw = bw / self.model.input.shape[1]
            bh = bh / self.model.input.shape[2]
            # Coordinates
            # Top left (x1, y1)
            # Bottom right (x2, y2)
            x1 = (bx - (bw / 2)) * image_size[1]
            y1 = (by - (bh / 2)) * image_size[0]
            x2 = (bx + (bw / 2)) * image_size[1]
            y2 = (by + (bh / 2)) * image_size[0] 
            boxes[i][..., 0] = x1 
            boxes[i][..., 1] = y1 
            boxes[i][..., 2] = x2
            boxes[i][..., 3] = y2    
        return boxes, box_confidences, box_class_probs
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        
        box_scores = []
        box_classes = []
        box_class_scores = []
        scores = []
        classes = []
        boxis = []
        for i in range(len(boxes)):
            box_scores.append(box_confidences[i] * box_class_probs[i])
            box_classes.append(np.argmax(box_scores[i], axis=-1))
            box_class_scores.append(np.max(box_scores[i], axis=-1))
            filtering_mask = box_class_scores[i] >= self.class_t
            scores += (box_class_scores[i][filtering_mask].tolist())
            boxis += (boxes[i][filtering_mask].tolist())
            classes += (box_classes[i][filtering_mask].tolist())
        scores = np.array(scores)
        boxis = np.array(boxis)
        classes = np.array(classes)
        return boxis, classes, scores
    
    
      


    
   