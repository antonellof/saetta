# saetta: C engine with Metal kernels. `make` builds $(BUILD)/bin/saetta, which runs
# Qwen3.8-Flash-Next and Qwen3.8-27B (each model's code compiled against its own constants,
# src/config_flash.h and src/config_27b.h; src/model_api.c picks one by the pack).
CC      := clang
BUILD   ?= build
GOLDEN  ?= tests/golden
# the 27B goldens next to GOLDEN (each names its pack in manifest.json); absent ones are skipped
GOLDENS_27B := golden-27b-4bit golden-27b-8bit golden-27b-4bit-pi golden-27b-4bit-partial golden-27b-turbo
# the Coder variant's golden (its manifest names the pack without a folder); skipped when absent
MODELS ?= $(or $(SAETTA_MODELS),$(HOME)/Models)
CODER_PACK ?= $(MODELS)/mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX

CFLAGS  := -std=c23 -O3 -ffp-contract=off -mcpu=native -mmacosx-version-min=27.2 \
           -Wall -Wextra -Wno-unused-parameter
OBJCFLAGS := -fobjc-arc -O2 -mcpu=native -mmacosx-version-min=27.2 -Wall -Wextra -Wno-unused-parameter
DEPFLAGS = -MMD -MP
LDFLAGS := -mmacosx-version-min=27.2
LDLIBS  := -framework Metal -framework Foundation -framework Accelerate -lpthread

OBJ  := $(BUILD)/obj
BIN  := $(BUILD)/bin/saetta
SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:src/%.c=$(OBJ)/%.o) $(OBJ)/metal/shim.o
TEST_SRCS := $(wildcard src/test/*.c)
TEST_OBJS := $(TEST_SRCS:src/test/%.c=$(OBJ)/test/%.o)
LIB_OBJS  := $(filter-out $(OBJ)/main.o,$(OBJS))

.PHONY: all test verify verify-dsv4 bench clean
.DELETE_ON_ERROR:

all: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/bin/saetta-test: $(LIB_OBJS) $(TEST_OBJS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# #embed'ed kernels (src/kernels/*.metal) are recorded in the .d files by -MMD.
$(OBJ)/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ)/metal/shim.o: src/metal/shim.m
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -c $< -o $@

# Runs every test, or those whose name contains T (make test T=json). SAETTA_PACK passes
# through the environment.
test: $(BUILD)/bin/saetta-test
	$(BUILD)/bin/saetta-test $(T)

# Flash-Next against GOLDEN, the Coder variant against golden-coder, then Qwen3.8-27B against
# the goldens beside it that exist
verify: $(BIN)
	$(BIN) verify $(GOLDEN)
	@d="$(dir $(abspath $(GOLDEN)))golden-coder"; \
	  if [ -d "$$d" ] && [ -d "$(CODER_PACK)" ]; then echo "verify: $$d"; SAETTA_PACK="$(CODER_PACK)" $(BIN) verify "$$d" || exit 1; \
	  elif [ -d "$$d" ]; then echo "verify: no Coder pack at $(CODER_PACK), golden-coder skipped"; fi
	@for g in $(GOLDENS_27B); do d="$(dir $(abspath $(GOLDEN)))$$g"; \
	  if [ -d "$$d" ]; then echo "verify: $$d"; env -u SAETTA_PACK $(BIN) verify "$$d" || exit 1; fi; done

# DeepSeek-V4-Flash against its goldens next to GOLDEN: the forced per-layer check on the
# all-positions golden, then the functional check on the others (streamed experts, cache
# SAETTA_EXPERT_CACHE_GB, default 40). Skipped when the pack is absent. Not part of verify.
DSV4_PACK ?= $(HOME)/Models/deepseek-ai/DeepSeek-V4-Flash-0731
GOLDENS_DSV4 := golden-dsv4-short-decode-all golden-dsv4-short-decode golden-dsv4-long
verify-dsv4: $(BIN)
	@if [ ! -d "$(DSV4_PACK)" ]; then echo "verify-dsv4: no DeepSeek-V4-Flash pack at $(DSV4_PACK), skipped"; exit 0; fi; \
	for g in $(GOLDENS_DSV4); do d="$(dir $(abspath $(GOLDEN)))$$g"; \
	  if [ -d "$$d" ]; then echo "verify: $$d"; \
	    SAETTA_PACK="$(DSV4_PACK)" SAETTA_EXPERT_CACHE_GB=$${SAETTA_EXPERT_CACHE_GB:-40} $(BIN) verify "$$d" || exit 1; \
	  else echo "verify-dsv4: $$d absent, skipped"; fi; done

bench: $(BIN)
	$(BIN) qgen-bench

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d) $(TEST_OBJS:.o=.d)
