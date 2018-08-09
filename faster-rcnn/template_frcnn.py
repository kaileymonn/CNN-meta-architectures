#!/usr/bin/env python3
## ----------------------------------------------------------------- ##
## -------------------------- FASTER-RCNN -------------------------- ##
## ----------------------------------------------------------------- ##

from amb import CVflow, ARM, FS

# Required
def init():
    FS.Stream('input', '/path/to/images')
    FS.Stream('anchors', '/path/to/predefined_anchors')
    CVflow.DAG('rpn', 'frcnn_rpn.pb') # Region Proposal Network, includes feature_extractor
    ARM.JIT('nms', 'nms.h', 'nms.c') # Regressor
    ARM.JIT('map_scores', 'map_scores.h', 'map_scores.c') # Just a simple mapping function
    ARM.JIT('crop', 'crop.h', 'crop.c') # Crop feature map based on nms-ed region proposals
    CVflow.DAG('classifier', 'frcnn_pool_and_classify.pb') # Foreground/background classifier and bounding box predictor

def loop():
                         
    img = next(FS.input)
    anchors = next(FS.anchors)
    
    ##########################################################################################################
    # -------------------------------------------- RPN layer ----------------------------------------------- #
    # Placeholder: 'Preprocessor/sub'                                                                        #
    #   'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_4e/concat' (?, ?, ?, 576) -> features_map  #
    #   'FirstStageBoxPredictor/Reshape_1 (?, ?, 2) -> region_proposals                                      #
    #   'FirstStageBoxPredictor/concat (?, ?, 1, 4) -> scores                                                #
    ##########################################################################################################
    feature_map, region_proposals, scores = CVflow.rpn(anchors, img)

    # perform nms on region_proposals to get rois
    rois = ARM.nms(scores, region_proposals, len(scores))

    # re-map rois to respective scores (due to truncation of region_proposals)
    mapped_scores = ARM.map_scores(scores, rois, region_proposals)

    # preprocessing to generate input for roi pooling layer
    cropped_feature_maps = ARM.crop(rois, feature_map)
    
    ##########################################################################################################
    # ---------------------- ROI pooling followed by classification and box-prediction --------------------- #
    # Placeholder: 'CropAndResize' (?, 14, 14, 576)                                                          #  
    #   'Squeeze_3' (?, 90, 4) -> classes                                                                    #
    #   'Squeeze_2' (?, 91) -> box_predictions                                                               #
    ##########################################################################################################
    classes, box_predictions = CVflow.classifier(cropped_feature_maps)
    
    # more nms to refine bounding boxes
    bounding_boxes = ARM.nms(mapped_scores, box_predictions, len(mapped_scores))

    yield bounding_boxes, classes




