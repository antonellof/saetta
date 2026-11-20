# saetta: C engine with Metal kernels. One build per model: `make` builds
# $(BUILD)/bin/saetta (Qwen3.8-Flash-Next) and $(BUILD)/bin/saetta-27b (Qwen3.8-27B);
# MODEL=27b selects the 27B build for verify and test. Each binary hands a pack of the other
# architecture to its sibling.
CC      := clang
BUILD   ?= build
MODEL   ?= flash

CFLAGS  := -std=c23 -O3 -ffp-contract=off -mcpu=native -mmacosx-version-min=27.2 \
           -Wall -Wextra -Wno-unused-parameter
OBJCFLAGS := -fobjc-arc -O2 -mcpu=native -mmacosx-version-min=27.2 -Wall -Wextra -Wno-unused-parameter
DEPFLAGS = -MMD -MP
LDFLAGS := -mmacosx-version-min=27.2
LDLIBS  := -framework Metal -framework Foundation -framework Accelerate -lpthread

ifeq ($(MODEL),27b)
CFLAGS  += -DSAETTA_27B
OBJCFLAGS += -DSAETTA_27B
OBJ  := $(BUILD)/obj-27b
BIN  := $(BUILD)/bin/saetta-27b
GOLDEN ?= tests/golden-27b-4bit
else ifeq ($(MODEL),flash)
OBJ  := $(BUILD)/obj
BIN  := $(BUILD)/bin/saetta
GOLDEN ?= tests/golden
else
$(error MODEL must be flash or 27b)
endif
SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:src/%.c=$(OBJ)/%.o) $(OBJ)/metal/shim.o
TEST_SRCS := $(wildcard src/test/*.c)
TEST_OBJS := $(TEST_SRCS:src/test/%.c=$(OBJ)/test/%.o)
LIB_OBJS  := $(filter-out $(OBJ)/main.o,$(OBJS))

.PHONY: all test verify bench clean
.DELETE_ON_ERROR:

all: $(BIN)
ifeq ($(MODEL),flash)
	@$(MAKE) --no-print-directory MODEL=27b $(BUILD)/bin/saetta-27b
endif

$(BIN): $(OBJS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/bin/saetta-test$(if $(filter 27b,$(MODEL)),-27b): $(LIB_OBJS) $(TEST_OBJS)
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
TEST_BIN := $(BUILD)/bin/saetta-test$(if $(filter 27b,$(MODEL)),-27b)
test: $(TEST_BIN)
	$(TEST_BIN) $(T)

verify: $(BIN)
	$(BIN) verify $(GOLDEN)

bench: $(BIN)
	$(BIN) qgen-bench

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d) $(TEST_OBJS:.o=.d)
