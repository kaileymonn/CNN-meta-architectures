#!/usr/bin/env make

MODEL_PATH   = C:\Users\c-kywong\Documents\meta_architectures\makefile_generator/path/to/model
NN_NAME      = ffrcn
SUPERDAG     = ffrcn
PROTO        = tensorflow/inception_v1/inception_v1-frozen-opt.pb
IDRA_FILE    = $(DIAG_DIR)/inception_input_images.txt
IP_LAYER     = input # TODO
IDATA        = ILSVRC2012_val_00000550.bin
IDATA_DIR    = /cv1/sequence/data_set/imagenet/test-images/bin8_mean128/
OP_LAYER     = InceptionV1/Logits/Predictions/Softmax
ODATA_FILE   = inception_out.bin
ODATA_ADES   = inception_out.txt
UCODE_DIR    = /cv2/work/schilkunda/default/repo_pace_SL0_cv22
RUN_DIR      = /dump/dump200/schilkunda/cv2_tests/$(NN_NAME)_superdag
COMMON_MK    = ./tensorflow/include.mk
NN_PARSER    = $(TF_TOOL)
PARSER_OPTS  = $(TF_OPTS)
CNNGEN_OPTS  = --config coeff-force-fx16 --iquantized --ifiledataformat 1,0,7,0 --ishape 1,3,224,224VAS_OPTS       += 
IREG_PATH    = 
IREG_COUNT   = 100
REG_EXPACC   = 0.896
REG_TOPN     = 5
include $(COMMON_MK)