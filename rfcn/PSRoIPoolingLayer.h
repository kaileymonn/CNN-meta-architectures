#ifndef PSROIPOOLING_H_
#define PSROIPOOLING_H_
#include "blob.h"

/* Similar to setup() in Caffe. Called once at the beginning. */
void psroipooling_setup(int id, blob* bottom1, blob* bottom2, blob* top);

/* Similar to forward() in Caffe. Called once per forward pass. */
void psroipooling_forward(int id, blob* bottom1, blob* bottom2, blob* top);

/* Similar to reshape() in Caffe. Not implmented. */
void psroipooling_reshape(int id, blob* bottom1, blob* bottom2, blob* top);

#endif

