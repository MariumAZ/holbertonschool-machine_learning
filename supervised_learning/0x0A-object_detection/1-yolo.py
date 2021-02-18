#!/usr/bin/env python3
import numpy as np
import tensorflow.keras as K 

class Yolo():
    def __init__(self,model_path, classes_path, class_t, nms_t, anchors):
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
        return 1/(1+np.exp(-x))        
    def process_outputs(self, outputs, image_size):
        #initialize empty lists:
        boxes = []
        box_confidences = []
        box_class_probs = []

        for (i,output) in enumerate(outputs):

            boxes.append(output[...,0:4])
            box_confidences.append(self.sigmoid(output[...,4:5]))
            box_class_probs.append(self.sigmoid(output[...,5:]))

            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            #network_output_coor = output[...,0:4]
            #network_output_prob = output[...,4:5]
            #network_output_classes = output[...,5:]

            t_x = output[...,0]
            t_y = output[...,1]
            t_w = output[...,2]
            t_h = output[...,3]
           
            #bounding box coordinates(x,y):
            cx = np.indices((grid_h, grid_w, anchor_boxes))[1]
            cy = np.indices((grid_h, grid_w, anchor_boxes))[0]
            b_x = (self.sigmoid(t_x) + cx) / grid_w
            b_y = (self.sigmoid(t_y) + cy) / grid_h

            #extract anchors dimensions:
            #i = outputs.index(output)
            a_w = self.anchors[i,:,0]
            a_h = self.anchors[i,:,1]

            #bounding box coordinates(w,h):
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

            b_w = a_w * np.exp(t_w) / input_w
            b_h = a_h * np.exp(t_h) / input_h

            x1 = b_x - b_w / 2
            x2 = x1 + b_w
            y1 = b_y - b_h / 2
            y2 = y1 + b_h

            img_w = image_size[1]
            img_h = image_size[0]

            boxes[i][..., 0] = x1 * img_w
            boxes[i][..., 1] = y1 * img_h
            boxes[i][..., 2] = x2 * img_w
            boxes[i][..., 3] = y2 * img_h
            
            
            return boxes,box_class_probs,box_confidences



               
            
            

            
        