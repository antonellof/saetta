# saetta: C engine with Metal kernels. `make` builds $(BUILD)/bin/saetta.
CC      := clang
BUILD   ?= build
GOLDEN  ?= tests/golden

CFLAGS  := -std=c23 -O3 -ffp-contract=off -mcpu=native -mmacosx-version-min=27.2 \
           -Wall -Wextra -Wno-unused-parameter
OBJCFLAGS := -fobjc-arc -O2 -mcpu=native -mmacosx-version-min=27.2 -Wall -Wextra -Wno-unused-parameter
DEPFLAGS = -MMD -MP
LDFLAGS := -mmacosx-version-min=27.2
LDLIBS  := -framework Metal -framework Foundation -framework Accelerate -lpthread

OBJ  := $(BUILD)/obj
SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:src/%.c=$(OBJ)/%.o) $(OBJ)/metal/shim.o
TEST_SRCS := $(wildcard src/test/*.c)
TEST_OBJS := $(TEST_SRCS:src/test/%.c=$(OBJ)/test/%.o)
LIB_OBJS  := $(filter-out $(OBJ)/main.o,$(OBJS))

.PHONY: all test verify bench clean
.DELETE_ON_ERROR:

all: $(BUILD)/bin/saetta

$(BUILD)/bin/saetta: $(OBJS)
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

verify: $(BUILD)/bin/saetta
	$(BUILD)/bin/saetta verify $(GOLDEN)

bench: $(BUILD)/bin/saetta
	$(BUILD)/bin/saetta qgen-bench

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d) $(TEST_OBJS:.o=.d)
