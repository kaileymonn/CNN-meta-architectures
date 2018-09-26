#!/usr/bin/env make

# @!README
# This makefile can be used to customize building dags
# The following are possible target invocations:
#   make        - builds recipe for `default`. This inturn prepares recipe
#                 for data->build->run. This is most commonly used recipe mix
#                 This recipe uses system parser with system CnnGen lib.
#   make all    - default recipe ALONG WITH ability to build CnnGen from source code procured
#                 from repo source ssh://$(USER)@ambus-git.ambarella.com:29418/cvtools/manifest
#                 This recipe uses system parser with local CnnGen lib.
#   make all CNNGEN_ROOT=<valid CnnGen source code dir>
#               - default recipe ALONG WITH ability to build CnnGen passed down by
#                 the user using CNNGEN_ROOT command line argument.
#                 This will force include.mk to use newly built local CnnGen lib
#                 with installed system parser
#   make data   - only procure data artefacts required for dag building
#   make tool   - only build CnnGen from source code procured from repo source
#                 ssh://sboga@ambus-git.ambarella.com:29418/cvtools/manifest
#   make tool CNNGEN_ROOT=<valid CnnGen source code dir>
#               - only build CnnGen from source code passed down by the user
#                 using CNNGEN_ROOT command line argument
#                 This will force include.mk to use newly built local CnnGen lib.
#   make build USE_LOCAL_PARSER=1
#               - only build neural network described by $(PROTO_FILE) &
#                 $(MODEL_FILE). This invocation will use parser from source code dir.
#                 This depends on data: target to be run atleast once
#   make build  - only build neural network described by $(PROTO_FILE) &
#                 $(MODEL_FILE). This invocation will use parser found in system path.
#                 This depends on data: target to be run atleast once
#   make run    - only run ades vector processor simulator
#
# valid combinations are:
# ***NOTE: <> is path to valid CnnGen source directory
# -----------------------------------------------------------------------------------------
#                 |    local parser (USE_LOCAL_PARSER=1)       |  system parser           |
# -----------------------------------------------------------------------------------------
# local AmbaCnn   | make all USE_LOCAL_PARSER=1                | make all                 |
# (all|tool)      | make tool USE_LOCAL_PARSER=1               | make tool                |
# (CNNGEN_ROOT=<>)| make all CNNGEN_ROOT=<> USE_LOCAL_PARSER=1 | make all CNNGEN_ROOT=<>  |
#                 | make tool CNNGEN_ROOT=<> USE_LOCAL_PARSER=1| make tool CNNGEN_ROOT=<> |
# -----------------------------------------------------------------------------------------
# system AmbaCnn  | make USE_LOCAL_PARSER=1                    | make                     |
# -----------------------------------------------------------------------------------------
#
# available variables from this COMMON makefile
# VARS that *must* be overwritten by Makefile in a {dag} directory
#   DIAG_DIR    - directory in which remoteconfig is run
#   PARSERS_DIR - directory formed relative to $(DIAG_DIR)
#
#   NN_NAME     - neural network name used for code generation
#   SUPERDAG    - preferred name for the SuperDAG
#
#   MODELS_ROOT - global directory prefix that hosts all models and protos
#                 this will be prefixed with $(PROTO) & $(MODEL)
#   PROTO       - suffix from $(MODELS_ROOT) of the input prototxt to use
#
#   MODEL       - suffix from $(MODELS_ROOT) of the input caffemodel to use
#
#   USE_DRAGEN  - use gen_image_list.txt to generate DRA images list
#                 with this option $(IDATA_DIR) is REQUIRED option.
#                 usage: check caffe/lenet/Makefile.in
#   IDRA_FILE   - full path to the text file that lists images used for DRA
#                 usage: check caffe/alexnet/Makefile.in
#   ***NOTE: $(USE_DRAGEN) and $(IDRA_FILE) are MUTUALLY EXCLUSIVE
#
#   IP_LAYER    - input layer that feeds data to the neural network
#   IDATA       - input file that is fed in to the network by IP_LAYER
#                 this value will be "prefixed" with $(IDATA_DIR)
#   IDATA_DIR   - directory with the $(IDATA) file
#   OP_LAYER    - output layer that stores data from network
#   LABEL_CSV   - file that labels each image in data_set, look up with img file
#
#   CNNGEN_OPTS - options to be passed directly to the CnnGen library
#                 e.g. --config coeff-force-fx16 --ifiledataformat 1,0,5,0
#   VAS_OPTS    - options to be passed to vas compiler
#                 e.g. -no-ice-expansion
#
#   NN_PARSER   - neural network parser that needs to be invoked for the diag
#                 the available parsers are: {CAFFE_TOOL, TF_TOOL, ONNX_TOOL}
#   PARSER_OPTS - options to parse the $(PROTO) & $(MODEL) files
#                 the available parser opts are: {CAFFE_OPTS, TF_OPTS, ONNX_OPTS}

## evaluate arguments for autoconf and automake
DIAG_DIR     = @srcdir@
PARSERS_DIR ?= $(abspath @srcdir@/../../../parsers/parser)

## encode model paths present in Ambarella internal network
MODELS_ROOT ?= /cv1/sequence/nn_models
# encode input artefacts to be processed
PROTO       ?=
MODEL       ?=
# these must be suffixes after $(MODELS_ROOT) prefix
PROTO_FILE  ?= $(MODELS_ROOT)/$(PROTO)
MODEL_FILE  ?= $(MODELS_ROOT)/$(MODEL)

## encode output ambarella artefacts to be generated
NN_NAME     ?= lenet

## encode sinks & sources of data fed in to the parser, cnngen, and superdag_gen
# - {outputs}
#   runtime data outputs
OP_DIR      ?= ./outputs
OP_LAYER    ?= prob
#   build time inputs
OCOEFFS_DIR ?= $(OP_DIR)/weights
ODATA_FILE  ?= $(NN_NAME)_$(OP_LAYER).bin
# - {inputs}
#   DRA inputs generation
USE_DRAGEN  ?= 0
IDRA_COUNT  ?= 1
IDRA        ?= $(NN_NAME)_dra_images.txt
IDRA_FILE   ?= $(addprefix $(IP_DIR)/, $(IDRA))
#   Runtime data inputs
IP_DIR      ?= ./inputs
IP_LAYER    ?= data
IDATA       ?= ILSVRC2012_val_00000550.bin
IDATA_DIR   ?= /cv1/sequence/data_set/imagenet/test-images/bin8_no_mean
IDATA_FILE  ?= $(addprefix $(IDATA_DIR)/, $(IDATA))
LABEL_CSV   ?= /cv1/sequence/data_set/imagenet/test-images/ILSVRC2012_validation_ground_truth_1000.csv
#   Regression test configuration
#    ** IREG_PATH can either be a file with images list or a directory holding images
IREG_PATH   ?= $(IDATA_DIR)
IREG_COUNT  ?= 1000
REG_EXPACC  ?= 0.96
REG_TOPN    ?= 1
#   Unit test configuration - targeted for ADES
ODATA_ADES  ?= $(NN_NAME)_$(OP_LAYER).txt
ADES_TOPN   ?= 5

## {superdag_gen} - encode generic arguments
SUPERDAG    ?= lenet
UCODE_DIR   ?= /cv2s/work/sboga/default/cv2s_uCode
RUN_DIR     ?= /dump/dump200/$(USER)/$(NN_NAME)_superdag/

## {CnnGen} - encode generic arguments
# - to build CnnGen from source code
CNNGEN_LOCAL = $(abspath AmbaCnn)
CNNGEN_BUILD = $(abspath build)
CNNGEN_ROOT  = $(abspath $(DIAG_DIR)/../../../../CnnGen)
# - options that are 'passed through' to CnnGen by <framework>parser.py
CNNGEN_OPTS +=

## {vas} - encode generic arguments
VAS_OPTS    += -auto -v -summary -dvi

### declare available parsers and their options
## {caffeparser.py} - encode generic options
CAFFE_TOOL   = $(shell tv2 -which caffeparser.py)
CAFFE_OPTS   = --proto $(IP_DIR)/$(notdir $(PROTO))	\
			   --model $(IP_DIR)/$(notdir $(MODEL))	\
			   --inputimages $(IDRA_FILE)	\
			   --output $(NN_NAME)	\
			   --outputfolder $(OP_DIR)	\
			   --iquantized	\
			   $(CNNGEN_OPTS)

## {tensorflowparser.py} - encode generic options
TF_TOOL      = $(shell tv2 -which tfparser.py)
TF_OPTS      = --proto $(IP_DIR)/$(notdir $(PROTO))	\
			   --inputimages $(IDRA_FILE)	\
			   --iquantized	\
			   --ifiledataformat 1,0,7,0	\
			   --inputshape 1,3,224,224	\
			   --config coeff-force-fx16	\
			   --output $(NN_NAME)	\
			   --outputfolder $(OP_DIR)	\
			   --outputnode $(OP_LAYER)	\
			   $(CNNGEN_OPTS)

## {onnxparser.py} - encode generic options
ONNX_TOOL    = $(shell tv2 -which onnxparser.py)
ONNX_OPTS    = --model $(IP_DIR)/$(notdir $(MODEL))	\
			   --inputimages $(IDRA_FILE)	\
			   --output $(NN_NAME)	\
			   --outputfolder $(OP_DIR)	\
			   --iquantized	\
			   $(CNNGEN_OPTS)

### using CAFFE as the default parser.
NN_PARSER   ?= $(ONNX_TOOL)
# - setting up CAFFE options as the default parser options
PARSER_OPTS ?= $(ONNX_OPTS)

.PHONY: default all build data accuracy run tool bub_superdag pace_superdag clean config FORCE

default: data build run
all: data tool build run

## Targets to convert code from <framework> format to vas code
# <frameworks> can be caffe, tensorflow (tf), onnx, etc.
build: config
	@echo "$@:"
	@echo "===== Begin Parser & CnnGen ====="
	@echo "-----(info) invoking parser with output dir @ $(OP_DIR)"
	LOCAL_AMBACNN=$(abspath $(PYAMBACNN)) $(NN_PARSER) $(PARSER_OPTS)
	@echo "-----(info) invoking vas in output dir @ $(OP_DIR)"
	cd $(OP_DIR); vas $(NN_NAME).vas $(VAS_OPTS)
	@echo "===== Finish Parser & CnnGen ====="
	@echo ""

## Targets to run vector processor simulator - ADES - for quick tests
run:
	@echo "$@:"
	@echo "===== Begin ADES run ====="
	@echo "-----(info) running ADES"
	run_ades.py --vasbasename $(NN_NAME)	\
			--path $(OP_DIR)	\
			--inputvectorsbin $(IP_LAYER)=$(IDATA_FILE)	\
			--outputvectorstxt $(OP_LAYER)=$(ODATA_ADES)
	@echo "-----(info) evaluting ades prediction category"
	eval_pred_category.py --predout $(OP_DIR)/$(ODATA_ADES)	\
			--labelfile $(LABEL_CSV)	\
			--inputfilename $(IDATA_FILE)	\
			--topn $(ADES_TOPN)
	@echo "===== Finish ADES run ====="
	@echo ""

# Targets to prepare superdag for BOARD hardware
bub_superdag:
	@echo "$@:"
	@echo "===== Begin BOARD superdag generation ====="
	superdag_gen.py --name $(SUPERDAG)	\
				--cvtask $(NN_NAME)	\
				--inputs $(IP_LAYER)=$(IDATA_FILE)	\
				--outputs $(OP_LAYER)=$(ODATA_FILE)	\
				--vasdir $(OP_DIR)	\
				--fwbase $(UCODE_DIR)	\
				--rundir $(RUN_DIR)	\
				--board
	@echo "===== Finish BOARD superdag generation ====="
	@echo ""

# Targets to prepare superdag for PACE simulation
pace_superdag:
	@echo "$@:"
	@echo "===== Begin PACE superdag generation ====="
	superdag_gen.py --name $(SUPERDAG)	\
				--cvtask $(NN_NAME)	\
				--inputs $(IP_LAYER)=$(IDATA_FILE)	\
				--outputs $(OP_LAYER)=$(ODATA_FILE)	\
				--vasdir $(OP_DIR)	\
				--fwbase $(UCODE_DIR)	\
				--rundir $(RUN_DIR)
	@echo "===== Finish PACE superdag generation ====="
	@echo ""

# Targets to prepare data for CnnGen, vas & superdag_generator
data:
	@echo "$@:"
	@echo "===== Begin data copy ====="
	mkdir -p $(IP_DIR)
	@echo "-----(info) copying model file"
	cp -f $(MODEL_FILE) $(IP_DIR)
	@if [ -f "$(PROTO_FILE)" ]; then	\
		echo "-----(info) copying proto file";	\
		echo "cp -f $(PROTO_FILE) $(IP_DIR)";	\
		cp -f $(PROTO_FILE) $(IP_DIR);	\
	fi
	@echo "-----(info) copying input data file"
	cp -f $(IDATA_FILE) $(IP_DIR)
	@echo "-----(info) copying pre-created sideband files, if there are any"
	$(foreach sideband, $(wildcard $(DIAG_DIR)/*.json), cp -f $(sideband) $(IP_DIR); )
	@if [ -f $(IREG_PATH) ]; then	\
		echo "-----(info) copying ACCURACY images-list file";	\
		echo "cp $(IREG_PATH) $(IP_DIR)";	\
		cp -f $(IREG_PATH) $(IP_DIR);	\
	fi
ifeq ($(USE_DRAGEN), 1)
	@echo "-----(info) generating <$(IDRA_FILE)> file"
	gen_image_list.py --folder $(IDATA_DIR) --output $(IDRA_FILE) --numimages $(IDRA_COUNT)
else
	@echo "-----(info) copying input DRA images-list file"
	$(foreach dra, $(IDRA_FILE), cp -f $(dra) $(IP_DIR); )
endif
	@echo "===== Finish data copy ====="
	@echo ""

# Targets to perform determine accuracy of a network
accuracy: data
	@echo "$@:"
	@echo "===== Begin Regression tests ====="
ifeq ("$(wildcard $(IREG_PATH)/.)", "")
	$(eval IREG_PATH := $(IP_DIR)/$(notdir $(IREG_PATH)))
endif
	ades_regression.py --vasbasename $(NN_NAME)	\
		--path $(OP_DIR)	\
		--inputimages $(IREG_PATH)	\
		--numimages $(IREG_COUNT)	\
		--inputnode $(IP_LAYER)	\
		--outputnode $(OP_LAYER)	\
		--labelfile $(LABEL_CSV)	\
		--baseline $(REG_EXPACC)	\
		--topn $(REG_TOPN);	\
	echo $$? > regression_result;
	@echo "===== Finish Regression tests ====="


# Targets to build CnnGen from source code
# check for proper CNNGEN_ROOT directory input
$(CNNGEN_ROOT): FORCE
	@echo "$@:"
	@echo "===== Begin AmbaCnn check ====="
	@if [ "$(CNNGEN_ROOT)" != "CnnGen" ] && [ ! -d "$(CNNGEN_ROOT)/src" ]; then	\
		echo "-----(error) invalid CnnGen root directory passed as input";	\
		echo "----- use {repo init -u ssh://$(USER)@ambus-git.ambarella.com:29418/cvtools/manifest -b master -m cv2.xml}";	\
		{ exit 1; };	\
	fi
	@echo "-----(info) using CnnGen directory at $(abspath $(CNNGEN_ROOT)) to build"
	@echo "===== Finish AmbaCnn check ====="
	@echo ""

# user can pass a valid CNNGEN_ROOT from command line,
tool: $(CNNGEN_ROOT)
	@echo "$@:"
	@echo "===== Begin AmbaCnn build ====="
	@echo "-----(info) creating CnnGen from sources"
	mkdir -p $(CNNGEN_BUILD)
	$(eval CNNGEN_SRC := $(abspath $(CNNGEN_ROOT)/src))
	@echo "-----(info) configuring with remoteconfig in $(CNNGEN_SRC)"
	@cd $(CNNGEN_BUILD); remoteconfig $(CNNGEN_SRC)
	@echo "-----(info) building with $(MAKE) in $(abspath $(CNNGEN_BUILD))"
	$(MAKE) -j11 -C $(CNNGEN_BUILD) OPT=-O3 prefix=$(CNNGEN_LOCAL) --no-print-directory
	@echo "===== Finish AmbaCnn build ====="
	@echo ""

clean:
	@echo "$@:"
	@echo "===== Begin clean ====="
	rm -rf $(IP_DIR) $(OP_DIR)
	@if [ -d $(CNNGEN_BUILD) ]; then	\
		cd $(CNNGEN_BUILD);	\
		make clean;	\
		rm -rf $(CNNGEN_LOCAL);	\
	fi
	@echo "===== Finish clean ====="
	@echo ""

config: FORCE
	@echo "$@:"
	@echo "===== Begin configuring build ====="
	@echo "DIAG_DIR=$(DIAG_DIR)"
	@echo "PARSERS_DIR=$(PARSERS_DIR)"
	$(eval PYAMBACNN := $(shell [ -d "$(CNNGEN_LOCAL)" ] && echo "$(CNNGEN_LOCAL)/python/amba_cnn_py"))
	@echo "LOCAL_AMBACNN=$(PYAMBACNN)"
ifeq ("$(USE_LOCAL_PARSER)", "1")
	$(eval ONNX_TOOL = "$(PARSERS_DIR)/onnx/onnxparser.py")
	@chmod +x $(NN_PARSER)
endif
	@echo "ONNX_TOOL=$(ONNX_TOOL)"
	@echo "===== Finish configuring build ====="
	@echo ""

FORCE:
