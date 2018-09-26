# vp_interface usage for initializing 4D tensors

/* most of the time, you have to do this:
*/
input->data[n*C*H*W + c*H*W + h*W + w]
 
/* if you really really want to, you can do this:
*/
int16_t**** tensor_4d = (int16_t****) malloc(sizeof(int16_t***) * N);
int16_t*** tensor_3d = (int16_t***) malloc(sizeof(int16_t**) * N*C);
int16_t** tensor_2d = (int16_t**) malloc(sizeof(int16_t*) * N*C*H);
int16_t* tensor_1d = input->data;
size_t idx;
for(size_t n = 0; n < N; n++) {
    idx = n;
    tensor_4d[idx] = tensor_3d[idx*C];
    for(size_t c = 0; c < C; c++) {
        idx = n*C + c;
        tensor_3d[idx] = tensor_3d[idx*H];
        for(size_t h = 0; h < H; h++) {
            idx = n*C*H + c*H + h;
            tensor_2d[idx] = tensor_1d[idx*W];
        }
    }
}
// NOW you can use your good old 4D array syntax
register int16_t something = tensor_4d[4][3][2][1]; // assuming 4<N, 3<C, 2<H, 1<W
