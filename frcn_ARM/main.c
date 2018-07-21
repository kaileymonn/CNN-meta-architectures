#include <time.h>
#include <stdio.h>
#include <malloc.h>
#include <stdint.h>
#include <stdlib.h>
#include "blob.h"
#include "ProposalLayer.h"
#include "PSRoIPoolingLayer.h"

void read_bin(char* path, blob* output) {
    size_t length;
    size_t n = output->n;
    size_t c = output->c;
    size_t h = output->h;
    size_t w = output->w;
    length = n * c * h * w;
    size_t size = _sizeof(output->type);
    
    if((output->data = malloc(length * size)) == NULL)
        fprintf(stderr, "ERROR: Ran out of memory.\n");
    
    FILE* f;
    if((f = fopen(path, "rb")) == NULL)
        fprintf(stderr, "ERROR: Cannot open file \"%s\"\n", path);
    fread(output->data, size, length, f);
    fclose(f);
    
    return;
}

int main(int argc, char* argv[]) {
    blob rpn_cls_prob_reshape, rpn_bbox_pred, im_info;
    blob rfcn_cls, rfcn_bbox;
    blob rois, cls_score, bbox_pred_pre;
    
    // Initialization
    rpn_cls_prob_reshape.type = INT16;
    rpn_cls_prob_reshape.n = 1;
    rpn_cls_prob_reshape.c = 18;
    rpn_cls_prob_reshape.h = 24;
    rpn_cls_prob_reshape.w = 32;
    rpn_bbox_pred.type = INT8;
    rpn_bbox_pred.n = 1;
    rpn_bbox_pred.c = 36;
    rpn_bbox_pred.h = 24;
    rpn_bbox_pred.w = 32;
    im_info.type = UINT32;
    im_info.n = 1;
    im_info.c = 1;
    im_info.h = 1;
    im_info.w = 3;
    rfcn_cls.type = INT8;
    rfcn_cls.n = 1;
    rfcn_cls.c = (20+1)*7*7;
    rfcn_cls.h = 24;
    rfcn_cls.w = 32;
    rfcn_bbox.type = INT8;
    rfcn_bbox.n = 1;
    rfcn_bbox.c = 8*7*7;
    rfcn_bbox.h = 24;
    rfcn_bbox.w = 32;
    rois.type = UINT16;
    rois.c = 1;
    rois.h = 1;
    rois.w = 5;
    cls_score.type = FLOAT32;
    cls_score.c = 20+1;
    cls_score.h = 1;
    cls_score.w = 1;
    bbox_pred_pre.type = FLOAT32;
    bbox_pred_pre.c = 8;
    bbox_pred_pre.h = 1;
    bbox_pred_pre.w = 1;

    // Setup prior to entrance into infinite loop
    proposal_setup(
            0, 
            &rpn_cls_prob_reshape, &rpn_bbox_pred, &im_info,
            &rois
        );
    psroipooling_setup(
            0,
            &rfcn_cls, &rois,
            &cls_score
        );
    psroipooling_setup(
            1,
            &rfcn_bbox, &rois,
            &bbox_pred_pre
        );
    im_info.type = UINT32;
    im_info.data = malloc(3 * _sizeof(UINT32));
    uint32_t* im_info_data = im_info.data;
    im_info_data[0] = 375;
    im_info_data[1] = 500;
    float scaling = 1.0f;
    im_info_data[2] = *((uint32_t*)&scaling);

    // Should be an infinite loop during deployment
    /*
     * while(True) {
     *     while(!VP_output_ready) {};
     */

    // Timing logic
    struct timespec start, stop;
    clockid_t clk_id = CLOCK_MONOTONIC;
    if(clock_gettime(clk_id, &start) == -1)
        exit(EXIT_FAILURE);

    // Parse .bin files
    // Fetch output from VP during deployment
    read_bin("../py-rfcn/rfcn_out/rfcn_bbox.bin", &rfcn_bbox);
    read_bin("../py-rfcn/rfcn_out/rfcn_cls.bin", &rfcn_cls);
    read_bin("../py-rfcn/rfcn_out/rpn_cls_prob_reshape.bin", 
             &rpn_cls_prob_reshape);
    read_bin("../py-rfcn/rfcn_out/rpn_bbox_pred.bin", &rpn_bbox_pred);
   
    // Prepare layers
    proposal_reshape(
            0, 
            &rpn_cls_prob_reshape, &rpn_bbox_pred, &im_info,
            &rois
        );
    psroipooling_reshape(
            0,
            &rfcn_cls, &rois,
            &cls_score
        );
    psroipooling_reshape(
            1,
            &rfcn_bbox, &rois,
            &bbox_pred_pre
        );
    
    // Evoke layers
    proposal_forward(
            0, 
            &rpn_cls_prob_reshape, &rpn_bbox_pred, &im_info,
            &rois
        );
    psroipooling_forward(
            0,
            &rfcn_cls, &rois,
            &cls_score
        );
    psroipooling_forward(
            1,
            &rfcn_bbox, &rois,
            &bbox_pred_pre
        );

    // Apply softmax
    int num = cls_score.n;
    float* data = cls_score.data;
    float* bbox = bbox_pred_pre.data;
    for(int i = 0; i < num; i++) {
        double sum = 0.0;
        for(int class = 0; class < 20+1; class++)
            sum += data[i*(20+1) + class];
        for(int class = 0; class < 20+1; class++) {
            data[i*(20+1) + class] /= sum;
        }
    }
    
    /* Print results */
    // Initialization
    int16_t* idx_scores = malloc(num * 2 * sizeof(int16_t));
    int* proposals = malloc(num * 4 * sizeof(int));
    for(int i = 0; i < num; i++) {
        idx_scores[2*i+1] = i;
        proposals[4*i+0] = ((uint16_t*)rois.data)[4*i+0]; //bbox[8*i+4];
        proposals[4*i+1] = ((uint16_t*)rois.data)[4*i+1]; //bbox[8*i+5];
        proposals[4*i+2] = ((uint16_t*)rois.data)[4*i+2]; //bbox[8*i+6];
        proposals[4*i+3] = ((uint16_t*)rois.data)[4*i+3]; //bbox[8*i+7];
    }
    // Loop through each class
    for(int class = 1; class < 20+1; class++) {
        // Extract scores for the class
        for(int i = 0; i < num; i++)
            idx_scores[2*i+0] = data[i*(20+1) + class];

        // Perform nms
        bool* keep = nms(idx_scores, proposals, num);
        for(int i = 0; i < num; i++)
            if(keep[i] && idx_scores[2*i] > 0.3*INT16_MAX)
                printf("Class %d (conf:%f) -- (%d,%d,%d,%d)\n",
                       class, (float)idx_scores[2*i]/INT16_MAX,
                       proposals[4*idx_scores[2*i+1]+0],
                       proposals[4*idx_scores[2*i+1]+1],
                       proposals[4*idx_scores[2*i+1]+2],
                       proposals[4*idx_scores[2*i+1]+3]);
        
        // Free memory
        free(keep);
    }
    
    // Timing logic
    if(clock_gettime(clk_id, &stop) == -1)
        exit(EXIT_FAILURE);
    unsigned long time_spent = (stop.tv_sec - start.tv_sec) * 1000000000L
                             + (stop.tv_nsec - start.tv_nsec);
    printf("TIME ELAPSED: %ld ms\n", time_spent / 1000000L);

    // Free memory
    free(rpn_bbox_pred.data);
    free(rpn_cls_prob_reshape.data);
    free(rfcn_bbox.data);
    free(rfcn_cls.data);
    free(idx_scores);
    free(proposals);

    // End of while loop
    /*
     * } // should not reach here
     * return 0;
     */
    return 1;
}

