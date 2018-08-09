/*
 * Type interface for ARM-VP communication
 *   - Must use the included types for effective inference.
 *   - When multiple function declarations are provided, they must have the same
 *     functional behavior but different type interfaces are allowed. Here, 
 *     dynamic type casting is as fast as static type casting (but no casting
 *     remains the fastest.)
 *   - C types are only OK for ARM-ARM communication
 *   - Must use the included helper functions to allocate memory
 *
 * July 20, 2018
 */
#ifndef VP_INTERFACE_H_
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#define SIMD_ALIGNMENT 64

/* 8-bit unsigned fixed-point datatype
 */
typedef struct vp_tensor_ufix8 {
    const uint_fast8_t exp_offset;
    const size_t n, c, h, w;
    uint8_t* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_ufix8_t;

typedef struct vp_scalar_ufix8 {
    const uint_fast8_t exp_offset;
    uint8_t data;
} vp_scalar_ufix8_t;

#define vp_tensor_ufix8_input const vp_tensor_ufix8_t* const __restrict__
#define vp_scalar_ufix8_input const vp_scalar_ufix8_t* const __restrict__
#define vp_tensor_ufix8_output vp_tensor_ufix8_t* const __restrict__
#define vp_scalar_ufix8_output vp_scalar_ufix8_t* const __restrict__

vp_tensor_ufix8_t* vp_tensor_ufix8_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset)
{
    // malloc in a memory-aligned manner
    uint8_t* data;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, sizeof(uint8_t)*n*c*h*w))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_ufix8_t temp = {.n=n, .c=c, .h=h, .w=w,
                              .exp_offset=exp_offset, .data=data};
    
    // malloc space for returned struct
    vp_tensor_ufix8_t* result;
    if((result = malloc(sizeof(vp_tensor_ufix8_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_ufix8_t));
    return result;
}

vp_tensor_ufix8_t* vp_tensor_ufix8_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset, const uint8_t* const src)
{
    // malloc in a memory-aligned manner
    uint8_t* data;
    size_t size = sizeof(uint8_t)*n*c*h*w;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, size))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_ufix8_t temp = {.n=n, .c=c, .h=h, .w=w,
                              .exp_offset=exp_offset, .data=data};
    
    // malloc space for returned struct
    vp_tensor_ufix8_t* result;
    if((result = malloc(sizeof(vp_tensor_ufix8_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_ufix8_t));
    
    // initialization
    if(src == NULL)
        memset(result->data, 0, size);
    else
        memcpy(result->data, src, size);
    return result;
}

vp_scalar_ufix8_t* vp_scalar_ufix8_malloc(
        const uint_fast8_t exp_offset) 
{
    // malloc in a memory-aligned manner
    vp_scalar_ufix8_t temp = {.exp_offset=exp_offset, .data=0};
    
    // malloc space for returned struct
    vp_scalar_ufix8_t* result;
    if((result = malloc(sizeof(vp_scalar_ufix8_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_scalar_ufix8_t));
    return result;
}

vp_scalar_ufix8_t* vp_scalar_ufix8_calloc(
        const uint_fast8_t exp_offset,
        const uint8_t input) 
{
    // malloc in a memory-aligned manner
    vp_scalar_ufix8_t temp = {.exp_offset=exp_offset, .data=input};
    
    // malloc space for returned struct
    vp_scalar_ufix8_t* result;
    if((result = malloc(sizeof(vp_scalar_ufix8_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_scalar_ufix8_t));
    return result;
}
// end: 8-bit unsigned fixed-point datatype

/* 8-bit signed fixed-point datatype
 */
typedef struct vp_tensor_fix8 {
    const uint_fast8_t exp_offset;
    const size_t n, c, h, w;
    int8_t* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_fix8_t;

typedef struct vp_scalar_fix8 {
    const uint_fast8_t exp_offset;
    int8_t data;
} vp_scalar_fix8_t;

#define vp_tensor_fix8_input const vp_tensor_fix8_t* const __restrict__
#define vp_scalar_fix8_input const vp_scalar_fix8_t* const __restrict__
#define vp_tensor_fix8_output vp_tensor_fix8_t* const __restrict__
#define vp_scalar_fix8_output vp_scalar_fix8_t* const __restrict__

vp_tensor_fix8_t* vp_tensor_fix8_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset)
{
    // malloc in a memory-aligned manner
    int8_t* data;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, sizeof(int8_t)*n*c*h*w))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_fix8_t temp = {.n=n, .c=c, .h=h, .w=w,
                              .exp_offset=exp_offset, .data=data};
    
    // malloc space for returned struct
    vp_tensor_fix8_t* result;
    if((result = malloc(sizeof(vp_tensor_fix8_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_fix8_t));
    return result;
}

vp_tensor_fix8_t* vp_tensor_fix8_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset, const int8_t* const src)
{
    // malloc in a memory-aligned manner
    int8_t* data;
    size_t size = sizeof(int8_t)*n*c*h*w;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, size))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_fix8_t temp = {.n=n, .c=c, .h=h, .w=w,
                              .exp_offset=exp_offset, .data=data};
    
    // malloc space for returned struct
    vp_tensor_fix8_t* result;
    if((result = malloc(sizeof(vp_tensor_fix8_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_fix8_t));
    
    // initialization
    if(src == NULL)
        memset(result->data, 0, size);
    else
        memcpy(result->data, src, size);
    return result;
}

vp_scalar_fix8_t* vp_scalar_fix8_malloc(
        const uint_fast8_t exp_offset) 
{
    // malloc in a memory-aligned manner
    vp_scalar_fix8_t temp = {.exp_offset=exp_offset, .data=0};
    
    // malloc space for returned struct
    vp_scalar_fix8_t* result;
    if((result = malloc(sizeof(vp_scalar_fix8_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_scalar_fix8_t));
    return result;
}

vp_scalar_fix8_t* vp_scalar_fix8_calloc(
        const uint_fast8_t exp_offset,
        const int8_t input) 
{
    // malloc in a memory-aligned manner
    vp_scalar_fix8_t temp = {.exp_offset=exp_offset, .data=input};
    
    // malloc space for returned struct
    vp_scalar_fix8_t* result;
    if((result = malloc(sizeof(vp_scalar_fix8_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_scalar_fix8_t));
    return result;
}
// end: 8-bit signed fixed-point datatype

/* 16-bit unsigned fixed-point datatype
 */
typedef struct vp_tensor_ufix16 {
    const uint_fast8_t exp_offset;
    const size_t n, c, h, w;
    uint16_t* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_ufix16_t;

typedef struct vp_scalar_ufix16 {
    const uint_fast8_t exp_offset;
    uint16_t data;
} vp_scalar_ufix16_t;

#define vp_tensor_ufix16_input const vp_tensor_ufix16_t* const __restrict__
#define vp_scalar_ufix16_input const vp_scalar_ufix16_t* const __restrict__
#define vp_tensor_ufix16_output vp_tensor_ufix16_t* const __restrict__
#define vp_scalar_ufix16_output vp_scalar_ufix16_t* const __restrict__

vp_tensor_ufix16_t* vp_tensor_ufix16_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset)
{
    // malloc in a memory-aligned manner
    uint16_t* data;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, sizeof(uint16_t)*n*c*h*w))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_ufix16_t temp = {.n=n, .c=c, .h=h, .w=w,
                              .exp_offset=exp_offset, .data=data};
    
    // malloc space for returned struct
    vp_tensor_ufix16_t* result;
    if((result = malloc(sizeof(vp_tensor_ufix16_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_ufix16_t));
    return result;
}

vp_tensor_ufix16_t* vp_tensor_ufix16_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset, const uint16_t* const src)
{
    // malloc in a memory-aligned manner
    uint16_t* data;
    size_t size = sizeof(uint16_t)*n*c*h*w;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, size))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_ufix16_t temp = {.n=n, .c=c, .h=h, .w=w,
                              .exp_offset=exp_offset, .data=data};
    
    // malloc space for returned struct
    vp_tensor_ufix16_t* result;
    if((result = malloc(sizeof(vp_tensor_ufix16_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_ufix16_t));
    
    // initialization
    if(src == NULL)
        memset(result->data, 0, size);
    else
        memcpy(result->data, src, size);
    return result;
}

vp_scalar_ufix16_t* vp_scalar_ufix16_malloc(
        const uint_fast8_t exp_offset) 
{
    // malloc in a memory-aligned manner
    vp_scalar_ufix16_t temp = {.exp_offset=exp_offset, .data=0};
    
    // malloc space for returned struct
    vp_scalar_ufix16_t* result;
    if((result = malloc(sizeof(vp_scalar_ufix16_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_scalar_ufix16_t));
    return result;
}

vp_scalar_ufix16_t* vp_scalar_ufix16_calloc(
        const uint_fast8_t exp_offset,
        const uint16_t input) 
{
    // malloc in a memory-aligned manner
    vp_scalar_ufix16_t temp = {.exp_offset=exp_offset, .data=input};
    
    // malloc space for returned struct
    vp_scalar_ufix16_t* result;
    if((result = malloc(sizeof(vp_scalar_ufix16_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_scalar_ufix16_t));
    return result;
}
// end: 16-bit unsigned fixed-point datatype

/* 16-bit signed fixed-point datatype
 */
typedef struct vp_tensor_fix16 {
    const uint_fast8_t exp_offset;
    const size_t n, c, h, w;
    int16_t* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_fix16_t;

typedef struct vp_scalar_fix16 {
    const uint_fast8_t exp_offset;
    int16_t data;
} vp_scalar_fix16_t;

#define vp_tensor_fix16_input const vp_tensor_fix16_t* const __restrict__
#define vp_scalar_fix16_input const vp_scalar_fix16_t* const __restrict__
#define vp_tensor_fix16_output vp_tensor_fix16_t* const __restrict__
#define vp_scalar_fix16_output vp_scalar_fix16_t* const __restrict__

vp_tensor_fix16_t* vp_tensor_fix16_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset)
{
    // malloc in a memory-aligned manner
    int16_t* data;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, sizeof(int16_t)*n*c*h*w))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_fix16_t temp = {.n=n, .c=c, .h=h, .w=w,
                              .exp_offset=exp_offset, .data=data};
    
    // malloc space for returned struct
    vp_tensor_fix16_t* result;
    if((result = malloc(sizeof(vp_tensor_fix16_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_fix16_t));
    return result;
}

vp_tensor_fix16_t* vp_tensor_fix16_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset, const int16_t* const src)
{
    // malloc in a memory-aligned manner
    int16_t* data;
    size_t size = sizeof(int16_t)*n*c*h*w;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, size))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_fix16_t temp = {.n=n, .c=c, .h=h, .w=w,
                              .exp_offset=exp_offset, .data=data};
    
    // malloc space for returned struct
    vp_tensor_fix16_t* result;
    if((result = malloc(sizeof(vp_tensor_fix16_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_fix16_t));
    
    // initialization
    if(src == NULL)
        memset(result->data, 0, size);
    else
        memcpy(result->data, src, size);
    return result;
}

vp_scalar_fix16_t* vp_scalar_fix16_malloc(
        const uint_fast8_t exp_offset) 
{
    // malloc in a memory-aligned manner
    vp_scalar_fix16_t temp = {.exp_offset=exp_offset, .data=0};
    
    // malloc space for returned struct
    vp_scalar_fix16_t* result;
    if((result = malloc(sizeof(vp_scalar_fix16_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_scalar_fix16_t));
    return result;
}

vp_scalar_fix16_t* vp_scalar_fix16_calloc(
        const uint_fast8_t exp_offset,
        const int16_t input) 
{
    // malloc in a memory-aligned manner
    vp_scalar_fix16_t temp = {.exp_offset=exp_offset, .data=input};
    
    // malloc space for returned struct
    vp_scalar_fix16_t* result;
    if((result = malloc(sizeof(vp_scalar_fix16_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_scalar_fix16_t));
    return result;
}
// end: 16-bit signed fixed-point datatype

/* 32-bit signed floating-point datatype WITHOUT denormalization
 */
typedef struct vp_tensor_float32 {
    const size_t n, c, h, w;
    float* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_float32_t;

typedef struct vp_scalar_float32 {
    float data;
} vp_scalar_float32_t;

#define vp_tensor_float32_input const vp_tensor_float32_t* const __restrict__
#define vp_scalar_float32_input const vp_scalar_float32_t* const __restrict__
#define vp_tensor_float32_output vp_tensor_float32_t* const __restrict__
#define vp_scalar_float32_output vp_scalar_float32_t* const __restrict__

vp_tensor_float32_t* vp_tensor_float32_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w)
{
    // malloc in a memory-aligned manner
    float* data;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, sizeof(float)*n*c*h*w))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_float32_t temp = {.n=n, .c=c, .h=h, .w=w, .data=data};
    
    // malloc space for returned struct
    vp_tensor_float32_t* result;
    if((result = malloc(sizeof(vp_tensor_float32_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_float32_t));
    return result;
}

vp_tensor_float32_t* vp_tensor_float32_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const float* const src)
{
    // malloc in a memory-aligned manner
    float* data;
    size_t size = sizeof(float)*n*c*h*w;
    if(posix_memalign((void**)&data, SIMD_ALIGNMENT, size))
        return NULL;
    data = __builtin_assume_aligned(data, SIMD_ALIGNMENT);
    vp_tensor_float32_t temp = {.n=n, .c=c, .h=h, .w=w, .data=data};
    
    // malloc space for returned struct
    vp_tensor_float32_t* result;
    if((result = malloc(sizeof(vp_tensor_float32_t))) == NULL)
        return NULL;
    memcpy(result, &temp, sizeof(vp_tensor_float32_t));
    
    // initialization
    if(src == NULL)
        memset(result->data, 0, size);
    else
        memcpy(result->data, src, size);
    return result;
}

vp_scalar_float32_t* vp_scalar_float32_malloc()
{
    vp_scalar_float32_t* temp;
    temp = malloc(sizeof(vp_scalar_float32_t));
    return temp;
}

vp_scalar_float32_t* vp_scalar_float32_calloc(const float input)
{
    vp_scalar_float32_t* temp;
    temp = malloc(sizeof(vp_scalar_float32_t));
    temp->data = input;
    return temp;
}
// end: 32-bit signed floating-point datatype WITHOUT denormalization

#endif
