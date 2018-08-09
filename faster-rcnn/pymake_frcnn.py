#!/usr/bin/env python3
import os

# available variables from include.mk
# VARS that *must* be used and overwritten
#   DIAG_DIR    - directory in which remoteconfig is run
#   PARSERS_DIR - directory formed relative to $(DIAG_DIR)
#
#   NN_NAME     - neural network name used for code generation
#   SUPERDAG    - preferred name for the SuperDAG
#
#   MODELS_ROOT - global directory prefix that hosts all protobufs
#                 this will be prefixed with $(PROTO)
#   PROTO       - suffix from $(MODELS_ROOT) of the input protobuf to use
#
#   USE_DRAGEN  - use gen_image_list.txt to generate DRA images list
#                 with this option $(IDATA_DIR) is REQUIRED option.
#                 $(IDRA) is NOT REQUIRED option.
#   IDRA_FILE   - full path to the text file that lists images used for DRA
#   ***NOTE: USE_DRAGEN and IDRA_FILE are MUTUALLY EXCLUSIVE
#
#   IP_LAYER    - input layer that feeds data to the neural network
#   IDATA       - input file that is fed in to the network by IP_LAYER
#                 this value will be "prefixed" with IDATA_DIR
#   IDATA_DIR   - directory with the IDATA file
#   OP_LAYER    - output layer that stores data from network
#   LABEL_CSV   - file that labels each image in data_set, look up with img file
#
#   NN_PARSER   - neural network parser that needs to be invoked for the diag
#                 the available parsers are: {CAFFE_TOOL, TF_TOOL, ONNX_TOOL}
#   PARSER_OPTS - options to use to parse the $(PROTO) & $(MODEL) files
#                 the available parser opts are: {CAFFE_OPTS, TF_OPTS, ONNX_OPTS}
                

# makefile_gen() takes parameters '/path/to/model' and a dictionary of model-specific makefile inputs
def makefile_gen(path, **inputs):
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'Makefile'), 'w') as out:
        out.write('#!/usr/bin/env make\n')
        out.write('\nMODEL_PATH   = '+ cwd + path)
        out.write('\nNN_NAME      = '+ inputs['nn_name'])
        out.write('\nSUPERDAG     = '+ inputs['nn_name'])
        out.write('\nPROTO        = '+ inputs['proto'])
        if('idra' in inputs):
            out.write('\nIDRA_FILE    = '+ '$(DIAG_DIR)/'+ inputs['idra'])
        else:
            out.write('\nUSE_DRAGEN   = '+ '1')
        out.write('\nIP_LAYER     = '+ inputs['ip_layer'])
        out.write('\nIDATA        = '+ inputs['idata'])
        out.write('\nIDATA_DIR    = '+ inputs['idata_dir'])
        out.write('\nOP_LAYER     = '+ inputs['op_layer'])
        out.write('\nODATA_FILE   = '+ inputs['odata_file'])
        out.write('\nODATA_ADES   = '+ inputs['odata_ades'])
        out.write('\nUCODE_DIR    = '+ inputs['ucode_dir'])
        out.write('\nRUN_DIR      = '+ inputs['run_dir'])

        if(inputs['framework'] == 'tensorflow'):
            out.write('\nCOMMON_MK    = '+'./tensorflow/include.mk')
            out.write('\nNN_PARSER    = '+ '$(TF_TOOL)')
            out.write('\nPARSER_OPTS  = '+ '$(TF_OPTS)')
        elif(inputs['framework'] == 'caffe'):
            out.write('\nCOMMON_MK    = '+'./caffe/include.mk')           
            out.write('\nNN_PARSER    = '+ '$(CAFFE_TOOL)')
            out.write('\nPARSER_OPTS  = '+ '$(CAFFE_OPTS)')
        elif(inputs['framework'] == 'onyx'):
            out.write('\nCOMMON_MK    = '+'./onyx/include.mk')
            out.write('\nNN_PARSER    = '+ '$(ONYX_TOOL)')
            out.write('\nPARSER_OPTS  = '+ '$(ONYX_OPTS)')          
   
        out.write('\nCNNGEN_OPTS  = '+ inputs['cnngen_opts'])
        out.write('VAS_OPTS       += ' + inputs['vas_opts'])
        out.write('\nIREG_PATH    = '+ inputs['ireg_path'])
        out.write('\nIREG_COUNT   = '+ '{}'.format(inputs['ireg_count']))
        out.write('\nREG_EXPACC   = '+ '{}'.format(inputs['reg_expacc']))
        out.write('\nREG_TOPN     = '+ '{}'.format(inputs['reg_topn']))
        
        # include common makefile to use Make framework
        # {MUST BE LAST LINE}
        out.write('\ninclude $(COMMON_MK)')