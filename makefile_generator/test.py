#!/usr/bin/env python3
import os
from pymake import makefile_gen

# nn_name       - neural network name used for code generation
# framework     - chosen framework
# proto         - suffix from $(MODELS_ROOT) of the input protobuf to use
# idra          - text file that lists images used for DRA 
# ip_layer      - input layer that feeds data to the neural network
# idata         - input file that is fed in to the network by IP_LAYER
#                 this value will be "prefixed" with IDATA_DIR
# idata_dir     - directory with the IDATA file
# op_layer      - output layer that stores data from network
# odata_file    - file that stores output data
# odata_ades    - file that stores ades data
# ucode_dir     - directory containing user code
# run_dir       - direction to run Makefile in
# cnngen_opts   - enforce CnnGen to use fixed16 weights && data-format 1,0,7,0
# vas_opts      - enforce vas options
# ireg_path     - accuracy check option
# ireg_count    - accuracy check option
# reg_expacc    - accuracy check option
# reg_topn      - accuracy check option

inputs = {
    'nn_name': 'ffrcn', 
    'framework': 'tensorflow',
    'proto': 'tensorflow/inception_v1/inception_v1-frozen-opt.pb',
    'idra': 'inception_input_images.txt', 
    'ip_layer': 'input # TODO',
    'idata': 'ILSVRC2012_val_00000550.bin',
    'idata_dir': '/cv1/sequence/data_set/imagenet/test-images/bin8_mean128/',
    'op_layer': 'InceptionV1/Logits/Predictions/Softmax',
    'odata_file': 'inception_out.bin',
    'odata_ades': 'inception_out.txt', 
    'ucode_dir': '/cv2/work/schilkunda/default/repo_pace_SL0_cv22',
    'run_dir': '/dump/dump200/schilkunda/cv2_tests/$(NN_NAME)_superdag',
    'cnngen_opts': '--config coeff-force-fx16 --iquantized --ifiledataformat 1,0,7,0 --ishape 1,3,224,224',
    'vas_opts': '',
    'ireg_path': '',
    'ireg_count': 100,
    'reg_expacc': 0.896,
    'reg_topn': 5,
}

makefile_gen('/path/to/model', **inputs)
