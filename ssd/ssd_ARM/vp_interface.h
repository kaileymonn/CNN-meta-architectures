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
/* 
 * Five datatypes are supported:
 *   - ufix8 (8-bit unsigned fixed-point datatype)
 *   - fix8 (8-bit signed fixed-point datatype)
 *   - ufix16 (16-bit unsigned fixed-point datatype)
 *   - fix16 (16-bit signed fixed-point datatype)
 *   - float32 (32-bit signed floating-point datatype)
 * With each of these datatypes, denoted as *dtype*, there are:
 *   - 2 struct definitions
 *   - 4 type aliases
 *   - 4 malloc/calloc functions
 * Specifically, for fixed-point datatypes, they are:
 *   - Struct definitions:
 *       - typedef struct vp_tensor_dtype {
 *             const uint_fast8_t exp_offset;
 *             const size_t n, c, h, w;
 *             dtype* const data; // aligned to 64 bytes
 *         } vp_tensor_dtype_t;
 *       - typedef struct vp_scalar_dtype {
 *             const uint_fast8_t exp_offset;
 *             dtype const data; // not aligned
 *         } vp_scalar_dtype_t;
 *   - Type aliases: (where '<>' can be 'tensor' or 'scalar')
 *       - typedef const vp_<>_dtype_t* const restrict vp_<>_dtype_input;
 *       - typedef vp_<>_dtype_t* restrict vp_<>_dtype_output;
 *   - Malloc functions:
 *       - vp_tensor_dtype_t* vp_tensor_dtype_malloc(
               const size_t n, const size_t c, const size_t h, const size_t w,
               const uint_fast8_t exp_offset
           );
 *       - vp_tensor_dtype_t* vp_tensor_dtype_malloc(,
               const uint_fast8_t exp_offset
           );
 *   - Calloc functions:
 *       - vp_tensor_dtype_t* vp_tensor_dtype_malloc(
               const size_t n, const size_t c, const size_t h, const size_t w,
               const uint_fast8_t exp_offset,
               const dtype* const src // source for initialization
           );
 *       - vp_tensor_dtype_t* vp_tensor_dtype_malloc(,
               const uint_fast8_t exp_offset,
               const type src // source for initialization
           );
 * Notes:
 *   - The interface for float32 is slightly different.
 *   - Calloc is slow, so please use it with caution.
 *   - Use vp_tensor_free or vp_scalar_free to prevent memory leak.
 *
 * July 24, 2018
 */
#ifndef VP_INTERFACE_H_
#include <stdint.h>
#define SIMD_ALIGNMENT    64
typedef enum state {
    uninitialized = 0, 
    valid,
    error
} state_t;

/* Aligned malloc and free
 * - Use only when necessary
 */
int _aligned_malloc(void** memptr, size_t alignment, size_t size);
void _aligned_free(void* ptr);
// end: Aligned malloc and free

/* 8-bit unsigned fixed-point datatype
 */
typedef struct vp_tensor_ufix8 {
    state_t status;
    const uint_fast8_t exp_offset;
    const size_t n, c, h, w;
    uint8_t* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_ufix8_t;

typedef struct vp_scalar_ufix8 {
    state_t status;
    const uint_fast8_t exp_offset;
    uint8_t data;
} vp_scalar_ufix8_t;

typedef const vp_tensor_ufix8_t* const __restrict__ vp_tensor_ufix8_input;
typedef const vp_scalar_ufix8_t* const __restrict__ vp_scalar_ufix8_input;
typedef vp_tensor_ufix8_t* __restrict__ vp_tensor_ufix8_output;
typedef vp_scalar_ufix8_t* __restrict__ vp_scalar_ufix8_output;

vp_tensor_ufix8_t* vp_tensor_ufix8_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset);
vp_tensor_ufix8_t* vp_tensor_ufix8_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset, const uint8_t* const src);
vp_scalar_ufix8_t* vp_scalar_ufix8_malloc(
        const uint_fast8_t exp_offset);
vp_scalar_ufix8_t* vp_scalar_ufix8_calloc(
        const uint_fast8_t exp_offset,
        const uint8_t input);
// end: 8-bit unsigned fixed-point datatype

/* 8-bit signed fixed-point datatype
 */
typedef struct vp_tensor_fix8 {
    state_t status;
    const uint_fast8_t exp_offset;
    const size_t n, c, h, w;
    int8_t* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_fix8_t;

typedef struct vp_scalar_fix8 {
    state_t status;
    const uint_fast8_t exp_offset;
    int8_t data;
} vp_scalar_fix8_t;

typedef const vp_tensor_fix8_t* const __restrict__ vp_tensor_fix8_input;
typedef const vp_scalar_fix8_t* const __restrict__ vp_scalar_fix8_input;
typedef vp_tensor_fix8_t* __restrict__ vp_tensor_fix8_output;
typedef vp_scalar_fix8_t* __restrict__ vp_scalar_fix8_output;

vp_tensor_fix8_t* vp_tensor_fix8_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset);
vp_tensor_fix8_t* vp_tensor_fix8_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset, const int8_t* const src);
vp_scalar_fix8_t* vp_scalar_fix8_malloc(
        const uint_fast8_t exp_offset);
vp_scalar_fix8_t* vp_scalar_fix8_calloc(
        const uint_fast8_t exp_offset,
        const int8_t input);
// end: 8-bit signed fixed-point datatype

/* 16-bit unsigned fixed-point datatype
 */
typedef struct vp_tensor_ufix16 {
    state_t status;
    const uint_fast8_t exp_offset;
    const size_t n, c, h, w;
    uint16_t* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_ufix16_t;

typedef struct vp_scalar_ufix16 {
    state_t status;
    const uint_fast8_t exp_offset;
    uint16_t data;
} vp_scalar_ufix16_t;

typedef const vp_tensor_ufix16_t* const __restrict__ vp_tensor_ufix16_input;
typedef const vp_scalar_ufix16_t* const __restrict__ vp_scalar_ufix16_input;
typedef vp_tensor_ufix16_t* __restrict__ vp_tensor_ufix16_output;
typedef vp_scalar_ufix16_t* __restrict__ vp_scalar_ufix16_output;

vp_tensor_ufix16_t* vp_tensor_ufix16_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset);
vp_tensor_ufix16_t* vp_tensor_ufix16_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset, const uint16_t* const src);
vp_scalar_ufix16_t* vp_scalar_ufix16_malloc(
        const uint_fast8_t exp_offset);
vp_scalar_ufix16_t* vp_scalar_ufix16_calloc(
        const uint_fast8_t exp_offset,
        const uint16_t input);
// end: 16-bit unsigned fixed-point datatype

/* 16-bit signed fixed-point datatype
 */
typedef struct vp_tensor_fix16 {
    state_t status;
    const uint_fast8_t exp_offset;
    const size_t n, c, h, w;
    int16_t* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_fix16_t;

typedef struct vp_scalar_fix16 {
    state_t status;
    const uint_fast8_t exp_offset;
    int16_t data;
} vp_scalar_fix16_t;

typedef const vp_tensor_fix16_t* const __restrict__ vp_tensor_fix16_input;
typedef const vp_scalar_fix16_t* const __restrict__ vp_scalar_fix16_input;
typedef vp_tensor_fix16_t* __restrict__ vp_tensor_fix16_output;
typedef vp_scalar_fix16_t* __restrict__ vp_scalar_fix16_output;

vp_tensor_fix16_t* vp_tensor_fix16_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset);
vp_tensor_fix16_t* vp_tensor_fix16_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const uint_fast8_t exp_offset, const int16_t* const src);
vp_scalar_fix16_t* vp_scalar_fix16_malloc(
        const uint_fast8_t exp_offset);
vp_scalar_fix16_t* vp_scalar_fix16_calloc(
        const uint_fast8_t exp_offset,
        const int16_t input);
// end: 16-bit signed fixed-point datatype

/* 32-bit signed floating-point datatype WITHOUT denormalization
 */
typedef struct vp_tensor_float32 {
    state_t status;
    const size_t n, c, h, w;
    float* const data __attribute__((__aligned__(SIMD_ALIGNMENT)));
} vp_tensor_float32_t;

typedef struct vp_scalar_float32 {
    state_t status;
    float data;
} vp_scalar_float32_t;

typedef const vp_tensor_float32_t* const __restrict__ vp_tensor_float32_input;
typedef const vp_scalar_float32_t* const __restrict__ vp_scalar_float32_input;
typedef vp_tensor_float32_t* __restrict__ vp_tensor_float32_output;
typedef vp_scalar_float32_t* __restrict__ vp_scalar_float32_output;

vp_tensor_float32_t* vp_tensor_float32_malloc(
        const size_t n, const size_t c, const size_t h, const size_t w);
vp_tensor_float32_t* vp_tensor_float32_calloc(
        const size_t n, const size_t c, const size_t h, const size_t w,
        const float* const src);
vp_scalar_float32_t* vp_scalar_float32_malloc();
vp_scalar_float32_t* vp_scalar_float32_calloc(const float input);
// end: 32-bit signed floating-point datatype WITHOUT denormalization

/* Free functions for all datatypes
 */
void vp_tensor_free(void* ptr);
void vp_scalar_free(void* ptr);
// end: Free functions for all datatypes

#endif
