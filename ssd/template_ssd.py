#!/usr/bin/env python3
## ----------------------------------------------------------------- ##
## ------------------------------ SSD ------------------------------ ##
## ----------------------------------------------------------------- ##

from amb import CVflow, ARM, FS

# Global variables
prev1, prev2 = None, None

# Required
def init():
    FS.Stream('input', '/path/to/images')
    CVflow.DAG('main_ssd', 'ssd_vgg_without_nms.pb') # Primary CNN for feature extraction, detection and classification
    ARM.JIT('nms', 'nms.h', 'nms.c') # Regressor

def loop():
    # data dependency graph:                            
    #                                                   
    # > SSD ---> NMS ------                         
    img = next(FS.input)
    
    # SSD model does most of the work, nms removes redundant priors
    class_detections = CVflow.main_ssd(img)
    predictions = ARM.nms(class_detections)
    yield predictions


