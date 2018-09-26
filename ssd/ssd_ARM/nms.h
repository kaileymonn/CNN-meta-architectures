#ifndef NMS_H_
#include "vp_interface.h"

inline vp_tensor_fix16_t* nms(vp_tensor_float32_input idx_scores, vp_tensor_fix16_input proposals, vp_scalar_fix16_input N);

#endif // NMS_H_