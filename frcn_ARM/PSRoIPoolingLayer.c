#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "blob.h"
#include "PSRoIPoolingLayer.h"

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

/* Global Variables */
const static int spatial_scale = 4; // 0.0625 == 2^-4
const static int pooled_height = 7;
const static int pooled_width = 7;

void psroipooling_setup(
        int id,
        blob* bottom1, blob* bottom2,
        blob* top) {
    assert(bottom2->c * bottom2->h * bottom2->w == 5);
    assert(pooled_height * pooled_width * top->c == bottom1->c);
    assert(1 == top->h);
    assert(1 == top->w);
    assert(bottom1->n == 1);
    return;
}

void psroipooling_forward(
        int id,
        blob* bottom1, blob* bottom2,
        blob* top) {
    size_t num = bottom2->n;
    size_t output_c = (bottom1->c / pooled_height) / pooled_width;
    size_t output_h = pooled_height;
    size_t output_w = pooled_width;
    int width = bottom1->w;
    int height = bottom1->h;

    // Extract data arrays
    top->n = bottom2->n;
    top->data = malloc(num * output_c * sizeof(float));
    uint16_t* rois = bottom2->data;
    int8_t* features = bottom1->data;
    float* scores = top->data;

    // Loop through each RoI
    for(size_t i = 0; i < num; i++) {
        int roi_batch_ind = rois[5*i];
        int roi_start_w = rois[5*i+1] >> spatial_scale;
        int roi_start_h = rois[5*i+2] >> spatial_scale;
        int roi_end_w = (rois[5*i+3] + 1) >> spatial_scale;
        int roi_end_h = (rois[5*i+4] + 1) >> spatial_scale;
        int roi_width = roi_end_w - roi_start_w;
        int roi_height = roi_end_h - roi_start_h;
        int roi_area = roi_width * roi_height;

        if(roi_width <= 0 || roi_height <= 0)
            continue;
        int bin_size_w = roi_width / pooled_width;
        int bin_size_h = roi_height / pooled_height;
        int bin_excess_w = roi_width % pooled_width;
        int bin_excess_h = roi_height % pooled_height;
        
        // Loop through each category
        for(size_t pc = 0; pc < output_c; pc++) {
            // Combine PSRoIPooling and (average) voting
            size_t score_idx = i * output_c + pc;
            double result = 0.0;

            // Loop through each bin
            for(size_t ph = 0; ph < output_h; ph++) {
                for(size_t pw = 0; pw < output_w; pw++) {
                    // Bin's area [wstart,wend) X [hstart,hend)
                    int wstart = (pw * bin_size_w + min(pw, bin_excess_w));
                    int hstart = (ph * bin_size_h + min(ph, bin_excess_h));
                    int wend = ((pw+1) * bin_size_w + min(pw+1, bin_excess_w));
                    int hend = ((ph+1) * bin_size_h + min(ph+1, bin_excess_h));
                    wstart = clamp(wstart + roi_start_w, width, 0);
                    hstart = clamp(hstart + roi_start_h, height, 0);
                    wend = clamp(wend + roi_start_w, width, 0);
                    hend = clamp(hend + roi_start_h, height, 0);

                    // Sum over the bin
                    size_t feature_idx = pc * (pooled_width * pooled_height
                                               * width * height)
                                       + ph * (pooled_width * width * height)
                                       + pw * (width * height);
                    long int partial_sum = 0L;
                    for(size_t h = hstart; h < hend; h++) {
                        for(size_t w = wstart; w < wend; w++) {
                            size_t bin_idx = h * width + w;
                            partial_sum += features[feature_idx + bin_idx];
                        }
                    }

                    // Add to total sum
                    result += partial_sum / (double)roi_area;
                }
            }
            
            // Store to output
            scores[score_idx] = max(result, 0.0f); 
        }
    }
    
    return;
}

void psroipooling_reshape(
        int id,
        blob* bottom1, blob* bottom2,
        blob* top) {
    return;
}

