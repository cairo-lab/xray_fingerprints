MAKEFLAGS += --silent
CONDA ?= $(shell which conda)
JUPYTER ?= $(shell which jupyter)
PYTHON ?= $(shell which python)
PIP ?= $(shell which pip)
RSYNC ?= $(shell which rsync)
TAG ?= latest
PWD = $(shell pwd)

all: clean clean-py

.PHONY: all

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

.PHONY: clean-pyc

clean:
	rm -rf build dist *.egg-info

.PHONY: clean

notebook:
	$(JUPYTER) notebook ./bias_partition.ipynb

.PHONY: notebook

sync:
	$(RSYNC) \
		--exclude=__pycache__ \
		--exclude=.git \
		--exclude=.env \
		--exclude=.DS_Store \
		--exclude=.vscode \
		--exclude=misc \
		-av $(PWD) discovery:"$(REMOTE_PROJECT_DIR)/analysis/results_bias"

.PHONY: sync

environment.yaml:
	$(CONDA) env export > $(PWD)/environment.yaml

slurm-interactive:
	srun \
		-p gpuq \
		--gres=gpu:k80 \
		--pty /bin/bash

.PHONY: slurm-interactive
