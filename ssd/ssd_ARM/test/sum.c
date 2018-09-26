#include <assert.h>
#include <stdio.h>
#include "vp_interface.h"

void add(vp_tensor_fix16_input input1, vp_tensor_fix16_input input2,
         vp_tensor_fix16_output output)
{
    //assert(input1->n == input2->n);
    //assert(input1->c == input2->c);
    //assert(input1->h == input2->h);
    //assert(input1->w == input2->w);
    size_t size = input1->n * input1->c * input1->h * input1->w;
    int16_t* temp1 = __builtin_assume_aligned(input1, SIMD_ALIGNMENT);
    int16_t* temp2 = __builtin_assume_aligned(input2, SIMD_ALIGNMENT);
    int16_t* temp3 = __builtin_assume_aligned(output, SIMD_ALIGNMENT);
    for(int i = 0; i < size; i++)
        temp3[i] = temp1[i] + temp2[i];
}

void sum(vp_tensor_float32_input input, vp_scalar_float32_output output) {
    //assert(input->n == 1 && input->c == 1);
    size_t size = input->h * input->w;
    register float sum = 0.0;
    for(size_t i = 0; i < size; i++)
        sum += input->data[i];
    output->data = sum;
}

int main() {
    vp_tensor_float32_t* temp = vp_tensor_float32_malloc(1, 1, 3, 4);
    for(size_t i = 0; i < 12; i++)
        if(i < 10)
            temp->data[i] = 1;
        else
            temp->data[i] = 0;
    vp_scalar_float32_t* out = vp_scalar_float32_malloc();

    sum(temp, out);
    printf("Sum output: %f\n", out->data);

    const int16_t data1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    vp_tensor_fix16_t* temp1 = vp_tensor_fix16_calloc(1, 1, 2, 1, 0, &data1);
    const int16_t data2[8] = {3, 4, 5, 6, 7, 8, 9, 0};
    vp_tensor_fix16_t* temp2 = vp_tensor_fix16_calloc(1, 1, 2, 1, 0, &data2);
    vp_tensor_fix16_t* temp3 = vp_tensor_fix16_malloc(1, 1, 2, 1, 0);

    add(temp1, temp2, temp3);
    for(size_t i = 0; i < 2; i++)
        printf("Add output: %hd\n", temp3->data[i]);

    return 0;
}

