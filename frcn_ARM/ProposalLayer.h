#ifndef PROPOSAL_H_
#define PROPOSAL_H_
#include <stdbool.h>
#include "blob.h"

bool* nms(int16_t* restrict idx_scores, int* restrict proposals, int N);

/* Similar to setup() in Caffe. Called once at the beginning. */
void proposal_setup(int id, blob* bottom1, blob* bottom2, 
                    blob* bottom3, blob* top);

/* Similar to forward() in Caffe. Called once per forward pass. */
void proposal_forward(int id, blob* bottom1, blob* bottom2, 
                      blob* bottom3, blob* top);

/* Similar to reshape() in Caffe. Not implmented. */
void proposal_reshape(int id, blob* bottom1, blob* bottom2, 
                      blob* bottom3, blob* top);

#endif

