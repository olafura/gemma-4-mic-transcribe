# Environment variables passed via elixir_make
# ERTS_INCLUDE_DIR
# MIX_APP_PATH

# XLA extension installation location
XLA_EXTENSION_DIR = cache/xla_extension
XLA_EXTENSION_LIB = $(XLA_EXTENSION_DIR)/lib
XLA_INCLUDE_PATH = $(XLA_EXTENSION_DIR)/include

# Cache configuration
EXLA_CACHE_SO = cache/libexla.so
EXLA_CACHE_OBJ_DIR = cache/$(EXLA_VERSION)/objs

# Private configuration
EXLA_DIR = c_src/exla
PRIV_DIR = $(MIX_APP_PATH)/priv
EXLA_SO = $(PRIV_DIR)/libexla.so
EXLA_LIB_DIR = $(PRIV_DIR)/xla_extension/lib

# Link paths
XLA_EXTENSION_LIB_LINK_PATH = ../$(CWD_RELATIVE_TO_PRIV_PATH)/$(XLA_EXTENSION_LIB)
EXLA_CACHE_SO_LINK_PATH = $(CWD_RELATIVE_TO_PRIV_PATH)/$(EXLA_CACHE_SO)

.DEFAULT_GOAL := $(EXLA_SO)

# Build flags
#
# Note that XLA requires c++17, Fine as well
#
# We need to define NDEBUG to out-out of debug mode. XLA is precompiled with
# that flag. In EXLA we include some XLA headers that may have conditionals
# on NDEBUG. Crucially, the conditionals may be around struct fields and if
# we omit the flag, the struct fields will differ from what the precompiled
# library uses leading to incompatible ABI and nasty segfaults.
CFLAGS += -fPIC -I$(ERTS_INCLUDE_DIR) -I$(FINE_INCLUDE_DIR) -I$(XLA_INCLUDE_PATH) -Wall -Wno-sign-compare \
	 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment \
	 -std=c++17 -w \
	 -DNDEBUG

ifdef DEBUG
	CFLAGS += -g
else
	CFLAGS += -O3
endif

ifeq ($(MIX_ENV),prod)
	CFLAGS += -DEXLA_PROD
endif

NVCC = $(CXX)
NVCCFLAGS = $(CFLAGS)
LDFLAGS += -L$(XLA_EXTENSION_LIB) -lxla_extension -shared -fvisibility=hidden
EXLA_SKIP_CUDA =
XLA_SNAPSHOT = $(shell cat cache/xla_snapshot.txt 2> /dev/null)
# Local GPU ISA for HIP kernels; override to cross-compile.
EXLA_HIP_ARCH ?= $(shell $(ROCM_PATH)/bin/rocm_agent_enumerator 2> /dev/null | grep -v gfx000 | head -1)
ifeq ($(EXLA_HIP_ARCH),)
EXLA_HIP_ARCH = gfx1151
endif

ifdef EXLA_CPU_ONLY
EXLA_SKIP_CUDA = 1
endif

ifneq ($(filter rocm%,$(XLA_TARGET) $(EXLA_TARGET)),)
EXLA_SKIP_CUDA = 1
endif

ifneq ($(findstring rocm,$(XLA_ARCHIVE_PATH)),)
EXLA_SKIP_CUDA = 1
endif

ifneq ($(findstring rocm,$(XLA_ARCHIVE_URL)),)
EXLA_SKIP_CUDA = 1
endif

ifneq ($(findstring rocm,$(XLA_SNAPSHOT)),)
EXLA_SKIP_CUDA = 1
endif

ifeq ($(CROSSCOMPILE),)
	# Interrogate the system for local compilation
	UNAME_S = $(shell uname -s)

ifeq ($(EXLA_SKIP_CUDA),)
$(info EXLA_CPU_ONLY is not set, checking for nvcc availability)
	NVCC_RESULT := $(shell which nvcc 2> /dev/null)
	NVCC_TEST := $(notdir $(NVCC_RESULT))

	ifeq ($(NVCC_TEST),nvcc)
$(info CUDA is available.)
		NVCC = nvcc
		NVCCFLAGS = -Xcompiler "$(CFLAGS)" -DCUDA_ENABLED
		CFLAGS += -DCUDA_ENABLED -I$(dir $(NVCC_RESULT))../include
	else
$(info CUDA is not available.)
	endif
else
$(info EXLA CUDA helper objects disabled for this build)
endif
else
	# Determine settings for cross-compiled builds like for Nerves
	UNAME_S = Linux
endif

ifeq ($(UNAME_S), Darwin)
	# Xcode 26.4+ errors on XLA headers specializing std::is_signed.
	CFLAGS += -Wno-invalid-specialization
	LDFLAGS += -flat_namespace -undefined dynamic_lookup -rpath @loader_path/xla_extension/lib
else
	# Use a relative RPATH, so at runtime libexla.so looks for libxla_extension.so
	# in ./lib regardless of the absolute location. This way priv can be safely
	# packed into an Elixir release. Also, we use $$ to escape Makefile variable
	# and single quotes to escape shell variable
	LDFLAGS += -Wl,-rpath,'$$ORIGIN/xla_extension/lib'
endif

# Optional test dylib: registers qr_cpu_custom_call_f32_exla_alias -> same
# handler as qr_cpu_custom_call_f32. Built only when MIX_ENV=test.
TEST_PLUGIN_CC = c_src/exla_test/custom_calls.cc
TEST_PLUGIN_SO = $(PRIV_DIR)/test/exla_qr_alias.so

$(TEST_PLUGIN_SO): $(TEST_PLUGIN_CC) | $(XLA_EXTENSION_DIR)
	@ mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -shared $(TEST_PLUGIN_CC) -o $@ $(LDFLAGS)

EXLA_SO_DEPS = $(EXLA_CACHE_SO)
ifeq ($(MIX_ENV),test)
EXLA_SO_DEPS += $(TEST_PLUGIN_SO)
endif

$(EXLA_SO): $(EXLA_SO_DEPS)
	@ mkdir -p $(PRIV_DIR)
	@ mkdir -p $(PRIV_DIR)/xla_extension
	@ if [ "${MIX_BUILD_EMBEDDED}" = "true" ]; then \
		cp -a $(abspath $(XLA_EXTENSION_LIB)) $(EXLA_LIB_DIR) ; \
		cp -a $(abspath $(EXLA_CACHE_SO)) $(EXLA_SO) ; \
	else \
		ln -sf $(XLA_EXTENSION_LIB_LINK_PATH) $(EXLA_LIB_DIR) ; \
		ln -sf $(EXLA_CACHE_SO_LINK_PATH) $(EXLA_SO) ; \
	fi

SOURCES = $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_client.cc $(EXLA_DIR)/exla_mlir.cc $(EXLA_DIR)/ipc.cc
SOURCES += $(wildcard $(EXLA_DIR)/custom_calls/*.cc)
HEADERS = $(EXLA_DIR)/exla_mlir.h $(EXLA_DIR)/custom_calls/runtime_callback_bridge.h $(EXLA_DIR)/exla_client.h $(EXLA_DIR)/exla_nif_util.h $(EXLA_DIR)/exla_log_sink.h $(EXLA_DIR)/ipc.h
OBJECTS = $(patsubst $(EXLA_DIR)/%.cc,$(EXLA_CACHE_OBJ_DIR)/%.o,$(SOURCES)) $(EXLA_CACHE_OBJ_DIR)/exla_cuda.o

# Hand-written HIP kernels (int4 GEMV) are compiled with hipcc and linked into
# libexla.so when building against a ROCm XLA extension. XLA's generic codegen
# dequantizes int4 weights before the GEMM; these kernels read packed nibbles.
HIPCC ?= $(ROCM_PATH)/bin/hipcc
ROCM_PATH ?= /opt/rocm
HIP_SOURCES = $(wildcard $(EXLA_DIR)/custom_calls/*.hip)
HIP_OBJECTS = $(patsubst $(EXLA_DIR)/%.hip,$(EXLA_CACHE_OBJ_DIR)/%.hip.o,$(HIP_SOURCES))

ifneq ($(EXLA_SKIP_CUDA),)
ifneq ($(wildcard $(HIPCC)),)
OBJECTS += $(HIP_OBJECTS)
LDFLAGS += -L$(ROCM_PATH)/lib -lamdhip64
endif
endif

$(EXLA_CACHE_OBJ_DIR)/%.hip.o: $(EXLA_DIR)/%.hip $(HEADERS)
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)/custom_calls
	$(HIPCC) $(CFLAGS) --offload-arch=$(EXLA_HIP_ARCH) -fPIC -c $< -o $@

$(EXLA_CACHE_OBJ_DIR)/exla_cuda.o: $(EXLA_DIR)/exla_cuda.cc $(EXLA_DIR)/exla_cuda.h
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXLA_CACHE_OBJ_DIR)/%.o: $(EXLA_DIR)/%.cc $(HEADERS)
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)/custom_calls
	$(CXX) $(CFLAGS) -c $< -o $@

$(EXLA_CACHE_SO): $(XLA_EXTENSION_DIR) $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXLA_CACHE_SO) $(LDFLAGS)

clean:
	rm -rf cache
