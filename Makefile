
SHELL = bash
BUILD-DIR = build
PYTHON := $(shell which python)

all: cmake

$(BUILD-DIR):
	-mkdir build

cmake: $(BUILD-DIR)
	cd $(BUILD-DIR) && rm -rf *
	cd $(BUILD-DIR) &&	cmake \
				-DPYTHON_EXECUTABLE=$(PYTHON) \
				-DCMAKE_BUILD_TYPE=Release ..
	cd $(BUILD-DIR) && make -j install

clean:
	rm -rf build
