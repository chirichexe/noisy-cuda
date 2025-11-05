# Configuration
BUILD_DIR := build
SRC_DIR := src
INC_DIR := include
TARGET := $(BUILD_DIR)/main

# Source files (extendable in the future)
C_SOURCES := $(SRC_DIR)/main.c $(SRC_DIR)/options.c $(SRC_DIR)/perlin.c
CUDA_SOURCES := $(SRC_DIR)/cuda_kernel.cu

# Object files
C_OBJECTS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
CUDA_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))
OBJECTS := $(C_OBJECTS) $(CUDA_OBJECTS)

# Compilers and flags
CC := gcc
NVCC := nvcc
CFLAGS := -I$(INC_DIR)
NVCC_FLAGS := -I$(INC_DIR)

# CUDA architecture (if specified)
ifdef ARCH
    NVCC_FLAGS += -arch=$(ARCH)
endif

# Main target
all: $(TARGET)

# Build target - renamed to avoid conflict
compile: $(TARGET)

$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(NVCC) $^ -o $@

# Compile C files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) -c $< $(CFLAGS) -o $@

# Compile CUDA files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) -c $< $(NVCC_FLAGS) -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Info target - displays usage information
info:
	@echo "=== Makefile Usage ==="
	@echo ""
	@echo "Available commands:"
	@echo "  make           - Build the project (creates executable in $(BUILD_DIR)/)"
	@echo "  make compile   - Build the project"
	@echo "  make clean     - Remove build directory and all compiled files"
	@echo "  make info      - Show this help message"
	@echo ""
	@echo "Optional parameters:"
	@echo "  ARCH=sm_XX    - Specify CUDA architecture (e.g., ARCH=sm_70)"
	@echo ""
	@echo "Examples:"
	@echo "  make                      # Normal build"
	@echo "  make ARCH=sm_70           # Build with specific CUDA architecture"
	@echo "  make clean                # Clean build files"
	@echo ""
	@echo "Project structure:"
	@echo "  Source files:    $(SRC_DIR)/"
	@echo "  Header files:    $(INC_DIR)/"
	@echo "  Build output:    $(BUILD_DIR)/"
	@echo "  Final executable: $(TARGET)"

# Phony targets
.PHONY: all compile clean info
