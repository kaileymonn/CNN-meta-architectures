#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <immintrin.h>
#include "blob.h"
#include "ProposalLayer.h"

/* Util Macros */
#define clampf(num, max, min) (fmaxf(fminf(num, max), min))
#define min(a,b) ({ __typeof__ (a) _a = (a); \
                    __typeof__ (b) _b = (b); \
                    _a > _b ? _b : _a; })
#define max(a,b) ({ __typeof__ (a) _a = (a); \
                    __typeof__ (b) _b = (b); \
                    _a > _b ? _a : _b; })
#define clamp(num,max,min) ({ __typeof__ (num) _num = (num); \
                              __typeof__ (max) _max = (max); \
                              __typeof__ (min) _min = (min); \
                              _num > _max ? _max : _num < _min ? _min : _num;})

/* Quicksort algorithm */
// assumes little-endianness
#include <endian.h>
#if __BYTE_ORDER == __LITTLE_ENDIAN
    #define SORT_NAME     sorter
    #define SORT_TYPE     int32_t
    #define SORT_CMP(x,y) (((y<<16)>>16) - ((x<<16)>>16))
    #include "sort/sort.h"
#else 
    assert(0)
#endif

/* Global Constants */
#define NMS_THRESH 0.7f
#define PRE_NMS_TOP_N 6000U
#define POST_NMS_TOP_N 300U
#define MIN_SIZE 16U
static const int num_anchors = 9;
static const int feat_stride = 16;
static const int anchors[9][4] = {
    { -84,  -40,  99,  55},
    {-176,  -88, 191, 103},
    {-360, -184, 375, 199},
    { -56,  -56,  71,  71},
    {-120, -120, 135, 135},
    {-248, -248, 263, 263},
    { -36,  -80,  51,  95},
    { -80, -168,  95, 183},
    {-168, -344, 183, 359}
};

//#pragma omp declare simd
static float iou(int* restrict xmins, int* restrict ymins,
                        int* restrict xmaxs, int* restrict ymaxs, 
                        int* restrict areas, int i, int j) {
    int x1 = max(xmins[i], xmins[j]);
    int y1 = max(ymins[i], ymins[j]);
    int x2 = min(xmaxs[i], xmaxs[j]);
    int y2 = min(ymaxs[i], ymaxs[j]);
    int area1 = areas[i];
    int area2 = areas[j];

    int i_area = max(x2 - x1, 0) * max(y2 - y1, 0);

    int u_area = area1 + area2 - i_area;
    
    return clampf(((float)i_area)/((float)u_area), 1.0f, 0.0f);
}


inline bool* nms(int16_t* restrict idx_scores,
                 int* restrict proposals, int N) {
    int counter = 0;
    int* xmins = malloc(N * sizeof(int));
    int* xmaxs = malloc(N * sizeof(int));
    int* ymins = malloc(N * sizeof(int));
    int* ymaxs = malloc(N * sizeof(int));
    int* areas = malloc(N * sizeof(int));
    bool* keep = malloc(N * sizeof(bool));
   
    // Rearrange elements 
    for(size_t i = 0; i < N; i++) {
        uint16_t idx = idx_scores[i*2+1];
        xmins[counter] = proposals[idx*4+0];
        ymins[counter] = proposals[idx*4+1];
        xmaxs[counter] = proposals[idx*4+2];
        ymaxs[counter] = proposals[idx*4+3];
        counter++;
    }
    //#pragma omp simd
    for(size_t i = 0; i < N; i++)
        areas[i] = (xmaxs[i]-xmins[i]) * (ymaxs[i]-ymins[i]);
    //#pragma omp simd
    for(size_t i = 0; i < N; i++)
        keep[i] = true;

    // NMS main loop
    for(size_t i = 0; i < N; i++) {
        if(!keep[i])
            continue;
        //#pragma omp simd
        //#pragma vector aligned
        for(size_t j = i+1; j < N; j++) {
            if(!keep[j])
                continue;
            float iou_result = iou(xmins, ymins, xmaxs, ymaxs, areas, i, j);
            if(iou_result > NMS_THRESH)
                keep[j] = false;
        }
    }

    // Free alloc'ed memory
    free(xmins);
    free(xmaxs);
    free(ymins);
    free(ymaxs);
    free(areas);
    
    return keep;
}

void proposal_setup(
        int id, 
        blob* bottom1, blob* bottom2, blob* bottom3,
        blob* top) {
    // Nothing to do
    return;
}

void proposal_forward(
        int id, 
        blob* bottom1, blob* bottom2, blob* bottom3,
        blob* top) {
    size_t h = bottom1->h;
    size_t w = bottom1->w;
    size_t K = h * w;
    int16_t* scores = ((int16_t*) bottom1->data) + num_anchors*h*w;
    int8_t* bbox_delta = (int8_t*) bottom2->data;
    uint32_t im_info[3] = {0};
    memcpy(im_info, bottom3->data, 3 * _sizeof(UINT32));

    // Initialization    
    int16_t* indexed_scores = malloc((K*num_anchors*2) * sizeof(int16_t));
    int* proposals = malloc((K*num_anchors*4) * sizeof(int));

    //#pragma omp simd
    for(size_t i = 0; i < h; i++) {
        for(size_t j = 0; j < w; j++) {
            int shift[4] = {j*feat_stride, i*feat_stride,
                               j*feat_stride, i*feat_stride};
            for(size_t k = 0; k < num_anchors; k++) {
                const int* base = anchors[k];
                int anchor[4] = {base[0]+shift[0], base[1]+shift[1],
                                 base[2]+shift[2], base[3]+shift[3]};
                size_t index = i*w*num_anchors + j*num_anchors + k;

                // Implement bbox_transform
                int width = anchor[2] - anchor[1] + 1;
                int height = anchor[3] - anchor[1] + 1;
                int ctr_x = anchor[0] + width / 2;
                int ctr_y = anchor[1] + height / 2;
                int pred_ctr_x = ((bbox_delta[index*4+0]*width)>>6) + ctr_x;
                int pred_ctr_y = ((bbox_delta[index*4+1]*height)>>6) + ctr_y;
                int pred_w = ldexpf(1.0157477086f,bbox_delta[index*4+2])*width;
                int pred_h = ldexpf(1.0157477086f,bbox_delta[index*4+3])*height;
                int pred_box[4] = {pred_ctr_x - pred_w / 2,
                                   pred_ctr_y - pred_h / 2,
                                   pred_ctr_x + pred_w / 2,
                                   pred_ctr_y + pred_h / 2};

                // Implement clip_boxes
                int proposal[4] = {clamp(pred_box[0], im_info[1], 0),
                                   clamp(pred_box[1], im_info[0], 0),
                                   clamp(pred_box[2], im_info[1], 0),
                                   clamp(pred_box[3], im_info[0], 0)};

                // Implement _filter_boxes
                float scaling = *((float*)&im_info[2]);
                int min_size = MIN_SIZE * scaling;
                int ws = proposal[2] - proposal[0] + 1;
                int hs = proposal[3] - proposal[1] + 1;
                if(ws < min_size || hs < min_size) {
                    indexed_scores[index*2+0] = INT16_MIN;
                    indexed_scores[index*2+1] = index;
                    proposals[index*4+0] = 0;
                    proposals[index*4+1] = 0;
                    proposals[index*4+2] = 0;
                    proposals[index*4+3] = 0;
                }
                else {
                    indexed_scores[index*2+0] = scores[index];
                    indexed_scores[index*2+1] = index;
                    proposals[index*4+0] = proposal[0];
                    proposals[index*4+1] = proposal[1];
                    proposals[index*4+2] = proposal[2];
                    proposals[index*4+3] = proposal[3];
                }
            }
        }
    }

    //#pragma omp simd
    {
        sorter_quick_sort((int32_t*)indexed_scores, K*num_anchors);
    }

    // Non-maximum suppression
    size_t num_proposals = min(K*num_anchors, PRE_NMS_TOP_N);
    bool* keep = nms(indexed_scores, proposals, num_proposals);
    for(size_t i = 0; i < K*num_anchors; i++)
        if (i >= num_proposals || !keep[i])
            indexed_scores[2*i+0] = INT16_MIN;
    //#pragma omp simd
    {
        sorter_quick_sort((int32_t*)indexed_scores, K*num_anchors);
    }

    // Copy to result
    top->n = POST_NMS_TOP_N;
    top->data = malloc(5 * POST_NMS_TOP_N * _sizeof(top->type));
    uint16_t* result = top->data;    
    for(size_t i = 0; i < POST_NMS_TOP_N; i++) {
        int16_t idx = indexed_scores[2*i+1];
        result[5*i] = 0U;
        result[5*i+1] = proposals[4*idx+0];
        result[5*i+2] = proposals[4*idx+1];
        result[5*i+3] = proposals[4*idx+2];
        result[5*i+4] = proposals[4*idx+3];
    }

    free(indexed_scores);
    free(proposals);    
    free(keep);

    return;
}

void proposal_reshape(
        int id, 
        blob* bottom1, blob* bottom2, blob* bottom3,
        blob* top) {
    // Nothing to do
    return;
}

