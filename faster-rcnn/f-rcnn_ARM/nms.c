#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "vp_interface.h"
#include "map_scores.h"

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
#define NMS_THRESH 0.3f // Global IoU threshold
#define NUM_ANCHORS 9 // Number of anchor boxes per proposal region

// Intersection area over Union area
static float iou(vp_tensor_fix16_input xmins, vp_tensor_fix16_input ymins,
                 vp_tensor_fix16_input xmaxs, vp_tensor_fix16_input ymaxs,
                 vp_tensor_fix16_input areas, int16_t i, int16_t j) {
    register float out;
    int16_t* temp_xmins = xmins->data;
    int16_t* temp_xmaxs = xmaxs->data;
    int16_t* temp_ymins = ymins->data;
    int16_t* temp_ymaxs = ymaxs->data;
    int16_t* temp_areas = areas->data;

    int16_t x1 = MAX(temp_xmins[i], temp_xmins[j]);
    int16_t y1 = MAX(temp_ymins[i], temp_ymins[j]);
    int16_t x2 = MIN(temp_xmaxs[i], temp_xmaxs[j]);
    int16_t y2 = MIN(temp_ymaxs[i], temp_ymaxs[j]);
    int16_t area1 = temp_areas[i];
    int16_t area2 = temp_areas[j];

    // Compute intersection and union areas
    int16_t i_area = MAX(x2 - x1 + 1, 0) * MAX(y2 - y1 + 1, 0);
    int16_t u_area = area1 + area2 - i_area;
    
    out = CLAMPF(((float)i_area)/((float)u_area), 1.0f, 0.0f);
    printf("\nComparing entries %d and %d: i_area = %d, u_area = %d, iou = %f\n",i, j, i_area, u_area, out);
    return out;
}

// -------------------------------------------------------------------------------------------------------------------------------------------------- //
// vp_tensor_fix16_t* nms() takes vp_tensor_float32_t* idx_scores - 1-D array of scores corresponding to N proposals.                                 //  
//                                vp_tensor_fix16_t*   proposals  - 1-D array of proposal coordinates and corresponding ID in the order:              //  
//                                                                      [bottom-left(x, y), top-right(x, y), class ID, ...]                           //  
//                                                                      There will be N x 5 entries.                                                  //  
//                                vp_scalar_fix16_t* N            - Number of proposals.                                                              //                                                     
// as arguments, returns a        vp_tensor_fix16_t*              - proposals with redundancies (non-maximal scores for overlapping regions) removed. //      
// -------------------------------------------------------------------------------------------------------------------------------------------------- //

vp_tensor_fix16_t* nms(vp_tensor_float32_input idx_scores,    // Scores of each anchor proposal, N scores 
                       vp_tensor_fix16_input proposals,       // There will be 1 entry in idx_scores corresponding to each proposal (5 entries, 5th column is proposal ID)
                       vp_scalar_fix16_input N) {             // Number of proposals
    // Safety checks
    assert(N->data > 0);
    assert(proposals->h == N->data && proposals->w == 5);
    assert(idx_scores->w == N->data);
    
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
        // printf("Dims: (%d, %d), (%d, %d).   Class_ID = %d\n", xmins->data[i], ymins->data[i], xmaxs->data[i], ymaxs->data[i], proposals->data[i*5+4]);   
    }

    // Main NMS loops
    int16_t num_keep = N->data;             // Keeps track of the number of 1's in keep[]
    for(size_t i = 0; i < N->data; i++) {
        if(keep->data[i] != 1) continue;
        for(size_t j = i+1; j < N->data; j++) {
            if(i == j) continue;
            if(keep->data[j] != 1) continue;
            float iou_result = iou(xmins, ymins, xmaxs, ymaxs, areas, i, j);
            if(iou_result >= NMS_THRESH) {
                // Exceeded IoU threshold, keep higher score of the 2 proposals
                printf("NMS threshold exceeded. iou value = %f\n", iou_result);
                num_keep--;
                if(idx_scores->data[i] >= idx_scores->data[j]) {
                    keep->data[j] = 0;
                    printf("idx_scores->data[proposal %zd] = %f > %f = idx_scores->data[proposal %zd]\n", i, idx_scores->data[i], idx_scores->data[j], j);
                    printf("Setting keep[proposal %zd] to 0\n", j);
                }
                else {
                    keep->data[i] = 0;
                    printf("idx_scores->data[proposal %zd] = %f < %f = idx_scores->data[proposal %zd]\n", i, idx_scores->data[i], idx_scores->data[j], j);
                    printf("Setting keep[proposal %zd] to 0\n", i);
                    break;
                }
            }
            else printf("Did not exceed NMS threshold, iou value = %f\n", iou_result);
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

    // Free allocated memory
    vp_tensor_free(xmins);
    vp_tensor_free(ymins);
    vp_tensor_free(xmaxs);
    vp_tensor_free(ymaxs);
    vp_tensor_free(areas);
    vp_tensor_free(keep);
            
    return output;
}

/* -----------------------------------------------------------------------
------------------------------- Testing ----------------------------------
----------------------------------------------------------------------- */
int main() {
    // Test nms
    #define NUM_PROPOSALS 6
    const float data_scores_1[NUM_PROPOSALS] = {0.5, 0.7, 0.88, 0.3, 0.66, 0.9};
    const int16_t data_proposals_1[NUM_PROPOSALS * 5] = {
        12, 84, 140, 212, 0, 
        24, 84, 152, 212, 0,
        36, 84, 164, 212, 0,
        12, 96, 140, 224, 1,
        24, 96, 152, 224, 1,
        24, 108, 152, 236, 2
    }; 

    #define NUM_PROPS_2 4
    const float data_scores_2[NUM_PROPOSALS] = {0.5, 0.7, 0.7, 0.65};
    const int16_t data_proposals_2[NUM_PROPOSALS * 5] = {
        12, 30, 76, 94, 0,
	    12, 36, 76, 100, 0,
	    72, 36, 200, 164, 1,
	    84, 48, 212, 176, 1
    };

    vp_scalar_fix16_t* N = vp_scalar_fix16_calloc(0, NUM_PROPOSALS);
    vp_tensor_float32_t* idx_scores_1 = vp_tensor_float32_calloc(1, 1, 1, N->data, &data_scores_1); 
    vp_tensor_fix16_t* proposals_1 = vp_tensor_fix16_calloc(1, 1, N->data, 5, 0, &data_proposals_1);
    vp_tensor_fix16_t* output_1 = nms(idx_scores_1, proposals_1, N);
    
    // Check output size
    assert(output_1->h == 1);
    
    printf("\nWe had %d proposals at first, now we have %zd proposals. NMS_THRESH = %f\n\n", NUM_PROPOSALS, output_1->h, NMS_THRESH);
    for(size_t i = 0; i < output_1->h; i++) {
        printf("Dims: (%d, %d), (%d, %d).   Class_ID = %d\n", output_1->data[i*5+0], output_1->data[i*5+1], output_1->data[i*5+2], output_1->data[i*5+3], output_1->data[i*5+4]);
    }

    N = vp_scalar_fix16_calloc(0, NUM_PROPS_2);
    vp_tensor_float32_t* idx_scores_2 = vp_tensor_float32_calloc(1, 1, 1, N->data, &data_scores_2); 
    vp_tensor_fix16_t* proposals_2 = vp_tensor_fix16_calloc(1, 1, N->data, 5, 0, &data_proposals_2);
    vp_tensor_fix16_t* output_2 = nms(idx_scores_2, proposals_2, N);
    
    // Check output size
    assert(output_2->h == 2);
    
    printf("\nWe had %d proposals at first, now we have %zd proposals. NMS_THRESH = %f\n\n", NUM_PROPS_2, output_2->h, NMS_THRESH);
    for(size_t i = 0; i < output_2->h; i++) {
        printf("Dims: (%d, %d), (%d, %d).   Class_ID = %d\n", output_2->data[i*5+0], output_2->data[i*5+1], output_2->data[i*5+2], output_2->data[i*5+3], output_2->data[i*5+4]);
    }
    
    // Compare mapped scores output  
    vp_tensor_float32_t* mapped_scores = map_scores(idx_scores_2, output_2, proposals_2);    
    for(size_t i = 0; i < idx_scores_2->w; i++) {
        printf("idx_scores: ");
        printf("%f", idx_scores_2->data[i]);
        printf("\n");
    }
    for(size_t i = 0; i < mapped_scores->w; i++) {
        printf("mapped_scores: ");
        printf("%f", mapped_scores->data[i]);
        printf("\n");     
    }

    printf("\nDon't compile on a windows machine, posix is a unix library.\n");

    // Free allocated memory
    vp_scalar_free(N);
    vp_tensor_free(idx_scores_1);
    vp_tensor_free(proposals_1);
    vp_tensor_free(output_1);
    vp_tensor_free(idx_scores_2);
    vp_tensor_free(proposals_2);
    vp_tensor_free(output_2);
    vp_tensor_free(mapped_scores);
    
    return 0;
}