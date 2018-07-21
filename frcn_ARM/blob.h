#ifndef BLOB_H_
#define BLOB_H_
#include <stdint.h>

enum dtype{INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32};

typedef struct blob_t {
    uint16_t n, c, h, w;
    enum dtype type;
    void* data;
} blob;

uint8_t _sizeof(enum dtype type);

#endif

