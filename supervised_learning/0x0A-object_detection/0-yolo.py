#!/usr/bin/env python3
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

        

        

        