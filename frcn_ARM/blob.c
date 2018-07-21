#include <stdint.h>
#include "blob.h"

uint8_t _sizeof(enum dtype type) {
    switch(type) {
        case UINT8:
        case INT8:
            return 1;
        case UINT16:
        case INT16:
            return 2;
        case UINT32:
        case INT32:
        case FLOAT32:
            return 4;
    }
}

void test() {
    return;
}

