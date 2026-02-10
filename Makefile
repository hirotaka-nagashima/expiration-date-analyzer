PYTHON_FILES := $(shell find src -name '*.py')

.PHONY: format lint check

format:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

lint:
	ruff check $(PYTHON_FILES)

check: format lint
