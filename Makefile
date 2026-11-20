# saetta: C engine with Metal kernels. `make` builds $(BUILD)/bin/saetta, which runs
# Qwen3.8-Flash-Next and Qwen3.8-27B (each model's code compiled against its own constants,
# src/config_flash.h and src/config_27b.h; src/model_api.c picks one by the pack).
CC      := clang
BUILD   ?= build
GOLDEN  ?= tests/golden
# the 27B goldens next to GOLDEN (each names its pack in manifest.json); absent ones are skipped
GOLDENS_27B := golden-27b-4bit golden-27b-4bit-pi

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

.PHONY: all test verify bench clean
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

# Flash-Next against GOLDEN, then Qwen3.8-27B against the goldens beside it that exist
verify: $(BIN)
	$(BIN) verify $(GOLDEN)
	@for g in $(GOLDENS_27B); do d="$(dir $(abspath $(GOLDEN)))$$g"; \
	  if [ -d "$$d" ]; then echo "verify: $$d"; env -u SAETTA_PACK $(BIN) verify "$$d" || exit 1; fi; done

bench: $(BIN)
	$(BIN) qgen-bench

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d) $(TEST_OBJS:.o=.d)
