#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "vp_interface.h"

/* ----------------------------------------------------------------------
--------------------------- Utility Macros ------------------------------
---------------------------------------------------------------------- */
#define MIN(a, b) ({ __typeof__ (a) _a = (a); \
                    __typeof__ (b) _b = (b); \
                    _a < _b ? _a : _b; })
#define MAX(a, b) ({ __typeof__ (a) _a = (a); \
                    __typeof__ (b) _b = (b); \
                    _a > _b ? _a : _b; })
#define CLAMPF(num, max, min) (fmaxf(fminf(num, max), min))
#define NMS_THRESH 0.1f // Global IoU threshold
#define NUM_ANCHORS 9 // Number of anchor boxes per proposal region

// Intersection area over Union area
static void iou(vp_tensor_fix16_input xmins, vp_tensor_fix16_input ymins,
                 vp_tensor_fix16_input xmaxs, vp_tensor_fix16_input ymaxs,
                 vp_tensor_fix16_input areas, int16_t i, int16_t j,
                 vp_scalar_float32_output out) {
    int16_t* temp_xmins = xmins->data;
    int16_t* temp_xmaxs = xmaxs->data;
    int16_t* temp_ymins = ymins->data;
    int16_t* temp_ymaxs = ymaxs->data;
    int16_t* temp_areas = areas->data;

    int16_t x1 = MAX(temp_xmins[i], temp_xmins[j]);
    int16_t y1 = MIN(temp_ymins[i], temp_ymins[j]);
    int16_t x2 = MIN(temp_xmaxs[i], temp_xmaxs[j]);
    int16_t y2 = MAX(temp_ymaxs[i], temp_ymaxs[j]);
    int16_t area1 = temp_areas[i];
    int16_t area2 = temp_areas[j];

    // Compute intersection and union areas
    int16_t i_area = MAX(x2 - x1, 0) * MAX(y1 - y2, 0);
    int16_t u_area = area1 + area2 - i_area;
    
    out->data = CLAMPF(((float)i_area)/((float)u_area), 1.0f, 0.0f);
    printf("\nComparing entries %d and %d: i_area = %d, u_area = %d, iou = %f\n",i, j, i_area, u_area, out->data);
}

// -------------------------------------------------------------------------------------------------------------------------------------------------- //
// vp_tensor_fix16_t* nms() takes vp_tensor_float32_t* idx_scores - 1-D array of scores corresponding to N proposals.                                 //  
//                                vp_tensor_fix16_t*   proposals  - 1-D array of proposal coordinates and corresponding ID in the order:              //  
//                                                                      [top-left(x, y), bottom-right(x, y), class ID, ...]                           //  
//                                                                      There will be N x 5 entries.                                                  //  
//                                vp_scalar_fix16_t* N            - Number of proposals.                                                              //                                                     
// as arguments, returns a        vp_tensor_fix16_t*              - proposals with redundancies (non-maximal scores for overlapping regions) removed. //      
// -------------------------------------------------------------------------------------------------------------------------------------------------- //

vp_tensor_fix16_t* nms(vp_tensor_float32_input idx_scores,    // Scores of each anchor proposal, N scores 
                       vp_tensor_fix16_input proposals,       // There will be 1 entry in idx_scores corresponding to each proposal (5 entries, 5th column is proposal ID)
                       vp_scalar_fix16_input N) {             // Number of proposals
    assert(N->data > 0);
    vp_tensor_fix16_t* xmins = vp_tensor_fix16_malloc(1, 1, 1, N->data, 0);
    vp_tensor_fix16_t* xmaxs = vp_tensor_fix16_malloc(1, 1, 1, N->data, 0);
    vp_tensor_fix16_t* ymins = vp_tensor_fix16_malloc(1, 1, 1, N->data, 0);
    vp_tensor_fix16_t* ymaxs = vp_tensor_fix16_malloc(1, 1, 1, N->data, 0);
    vp_tensor_fix16_t* areas = vp_tensor_fix16_malloc(1, 1, 1, N->data, 0);
    vp_tensor_fix16_t* keep = vp_tensor_fix16_malloc(1, 1, 1, N->data, 0);      // maps i-th proposal to keep[i] = 1 or 0 corresponding to 1: Keep proposal, 0: Discard proposal

    // Initialize array values
    for(size_t i = 0; i < N->data; i++) {
        xmins->data[i] = proposals->data[i*5+0];
        ymins->data[i] = proposals->data[i*5+1];
        xmaxs->data[i] = proposals->data[i*5+2];
        ymaxs->data[i] = proposals->data[i*5+3];
        areas->data[i] = abs(xmaxs->data[i] - xmins->data[i]) * abs(ymaxs->data[i] - ymins->data[i]);
        keep->data[i] = 1;
        printf("Dims: (%d, %d), (%d, %d).   Class_ID = %d\n", xmins->data[i], ymins->data[i], xmaxs->data[i], ymaxs->data[i], proposals->data[i*5+4]);   
    }

    // Main NMS loops
    int16_t num_keep = N->data;             // Keeps track of the number of 1's in keep[]
    for(size_t i = 0; i < N->data; i++) {
        if(keep->data[i] != 1){
            printf("Oh no\n");
            continue;
        } 
        for(size_t j = i+1; j < N->data; j++) {
            if(keep->data[j] != 1)
                continue;
            vp_scalar_float32_t* iou_result = vp_scalar_float32_malloc();
            iou(xmins, ymins, xmaxs, ymaxs, areas, i, j, iou_result);
            // printf("IoU for ID %d: %f\n", j, iou_result->data);
            if(iou_result->data > NMS_THRESH) {
                // Exceeded IoU threshold, keep higher score of the 2 proposals
                num_keep--;
                if(idx_scores->data[i] > idx_scores->data[j]) {
                    keep->data[j] = 0;
                    printf("Setting keep[proposal %zd] to 0\n", j);
                }
                else {
                    keep->data[i] = 0;
                    printf("Setting keep[proposal %zd] to 0\n", i);
                    break;
                }
                printf("NMS threshold exceeded. iou value = %f\n", iou_result->data);
            }
            else
                printf("Did not exceed NMS threshold, iou value = %f\n", iou_result->data);
        }
    }
    assert(num_keep <= N->data);

    // Re-generate proposals, discard redundant proposals
    int16_t idx = 0;
    vp_tensor_fix16_t* output = vp_tensor_fix16_malloc(1, 1, num_keep, 5, 0);    
    for(size_t i = 0; i < num_keep; i++) {
        // Find next i such that keep[i] = 1 
        while(keep->data[idx] != 1) {
            idx++;
        }
        printf("\nAdding proposal number: %d\n", idx);    
        output->data[i*5+0] = proposals->data[idx*5+0];
        output->data[i*5+1] = proposals->data[idx*5+1];
        output->data[i*5+2] = proposals->data[idx*5+2];
        output->data[i*5+3] = proposals->data[idx*5+3];
        output->data[i*5+4] = proposals->data[idx*5+4];
        idx++;
    }
    return output;
}

/* -----------------------------------------------------------------------
------------------------------- Testing ----------------------------------
----------------------------------------------------------------------- */
int main() {
    // Test IoU
    const int16_t data_xmins[2] = {3, 4};
    const int16_t data_xmaxs[2] = {6, 8};
    const int16_t data_ymins[2] = {10, 8};
    const int16_t data_ymaxs[2] = {6, 4};
    const int16_t data_areas[2] = {12, 16};
    
    vp_tensor_fix16_t* xmins = vp_tensor_fix16_calloc(1, 1, 1, 2, 0, &data_xmins);
    vp_tensor_fix16_t* xmaxs = vp_tensor_fix16_calloc(1, 1, 1, 2, 0, &data_xmaxs);
    vp_tensor_fix16_t* ymins = vp_tensor_fix16_calloc(1, 1, 1, 2, 0, &data_ymins);
    vp_tensor_fix16_t* ymaxs = vp_tensor_fix16_calloc(1, 1, 1, 2, 0, &data_ymaxs);
    vp_tensor_fix16_t* areas = vp_tensor_fix16_calloc(1, 1, 1, 2, 0, &data_areas);
    vp_scalar_float32_t* iou_output = vp_scalar_float32_malloc();

    iou(xmins, ymins, xmaxs, ymaxs, areas, 0, 1, iou_output);
    printf("IoU Output: %f\n", iou_output->data);

    // Test nms
    #define NUM_PROPOSALS 4
    const float data_scores[NUM_PROPOSALS] = {0.4, 0.85, 0.65, 0.77};
    const int16_t data_proposals[NUM_PROPOSALS * 5] = {
        3, 10, 6, 6, 0, 
        4, 8, 8, 4, 0,
        25, 35, 50, 23, 1,
        28, 38, 57, 26, 1
    }; 

    vp_scalar_fix16_t* N = vp_scalar_fix16_calloc(0, NUM_PROPOSALS);
    vp_tensor_float32_t* idx_scores = vp_tensor_float32_calloc(1, 1, N->data, 1, &data_scores); 
    vp_tensor_fix16_t* proposals = vp_tensor_fix16_calloc(1, 1, N->data, 5, 0, &data_proposals);
    vp_tensor_fix16_t* output = nms(idx_scores, proposals, N);
    
    printf("\nWe had %d proposals at first, now we have %zd proposals\n\n", NUM_PROPOSALS, output->h);
    for(size_t i = 0; i < output->h; i++) {
        printf("Dims: (%d, %d), (%d, %d).   Class_ID = %d\n", output->data[i*5+0], output->data[i*5+1], output->data[i*5+2], output->data[i*5+3], output->data[i*5+4]);
    }

    printf("\nDon't compile on a windows machine, posix is a unix library.\n");
    return 0;
}