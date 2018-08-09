#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "vp_interface.h"

// ------------------------------------------------------------------------------------------------------------------------------------------------------ //
// vp_tensor_fix16_t* crop() takes vp_tensor_fix16_t*   rois            - nms output with data array in the following format:                             //  
//                                                                          [bottom-left(x, y), top-right(x, y), class ID, ...]                           //
//                                 vp_tensor_float32_t* feature_map     - feature map generated for each image (flattened to 1-D array)                   //  
//                                                                          [feature_map of channel 0, feature_map of channel 1, ...]                     //
// as arguments, returns a         vp_tensor_float32_t* cropped_map     - 1-D array of n*c*roi_area entries                                               //      
// ------------------------------------------------------------------------------------------------------------------------------------------------------ //
vp_tensor_float32_output crop(vp_tensor_fix16_input rois,       
                              vp_tensor_float32_input feature_map) {
    // Safety checks
    assert(rois->h > 0);
    assert(feature_map->n == 1);

    int16_t num_rois = rois->h;
    int16_t map_area = feature_map->h*feature_map->w;
    int16_t offset = 0;
    vp_tensor_float32_t* cropped_feature_maps = vp_tensor_float32_malloc(num_rois, feature_map->c, feature_map->h, feature_map->w);
    printf("malloc-ed size: %d\n", num_rois*map_area*feature_map->c);
    for(size_t i = 0; i < num_rois; i++) {
        int16_t roi_rows = rois->data[i*5+3] - rois->data[i*5+1] + 1;
        int16_t roi_cols = rois->data[i*5+2] - rois->data[i*5+0] + 1;
        int16_t roi_area = roi_rows * roi_cols;
        printf("roi_area = %d\n", roi_area);
        for(size_t j = 0; j < feature_map->c; j++) {
            for(size_t k = 0; k < roi_rows; k++) {
                for(size_t l = 0; l < roi_cols; l++) {
                    const int16_t curr_row = rois->data[i*5+1] + k;
                    const int16_t curr_col = rois->data[i*5+0] + l;
                    cropped_feature_maps->data[offset + j*roi_area + k*roi_cols + l] = feature_map->data[j*map_area + feature_map->w*curr_row + curr_col];
                    // printf("feature_map_data_index: %d\n", j*map_area + feature_map->w*curr_row + curr_col);
                }
            }            
        }
        offset += roi_area*feature_map->c;
    }
    vp_tensor_float32_output process_map = vp_tensor_float32_calloc(1, 1, 1, offset, &cropped_feature_maps->data[0]);
    return process_map;
}

/* -----------------------------------------------------------------------
------------------------------- Testing ----------------------------------
----------------------------------------------------------------------- */
int main() {
    const float feature_map_scores[1*5*5] = {
        0.3, 0.4, 0.3, 0.4, 0.5,
        0.3, 0.4, 0.3, 0.4, 0.5,
        0.1, 0.2, 0.6, 0.8, 0.5,
        0.1, 0.2, 0.7, 0.9, 0.5,
        0.1, 0.2, 0.3, 0.4, 0.5
    };
    
    const int16_t rois_raw[5*2] = {
        0, 0, 2, 2, 0,
        2, 2, 4, 4, 1
    };
    vp_tensor_float32_t* feature_map = vp_tensor_float32_calloc(1, 1, 5, 5, &feature_map_scores);
    vp_tensor_fix16_t* rois = vp_tensor_fix16_calloc(1, 1, 2, 5, 0, &rois_raw);

    vp_tensor_float32_t* cropped_map = crop(rois, feature_map);
    for(int i = 0; i < cropped_map->w; i++) {
        if(i % 3 == 0) {printf("\n");}
        printf("%f ", cropped_map->data[i]);
    }
    printf("\n");

    // vp_tensor_float32_t** tensor_4d = vp_tensor_float32_malloc(1, 1, 1, cropped_map->w);
    // tensor_4d[0][0]->data = &cropped_map->data[9];

    vp_tensor_free(feature_map);
    vp_tensor_free(rois);
    vp_tensor_free(cropped_map);

    return 0;
}
