.PHONY: help init-train init-docker-train run-container jupyter test lint clean clean-model clean-pyc clean-data clean-train-docker clean-train-container clean-train-image sync-from-source sync-to-source
.DEFAULT_GOAL := help

###########################################################################################################
## SCRIPTS
###########################################################################################################

define PRINT_HELP_PYSCRIPT
import os, re, sys

if os.environ['TARGET']:
    target = os.environ['TARGET']
    is_in_target = False
    for line in sys.stdin:
        match = re.match(r'^(?P<target>{}):(?P<dependencies>.*)?## (?P<description>.*)$$'.format(target).format(target), line)
        if match:
            print("target: %-20s" % (match.group("target")))
            if "dependencies" in match.groupdict().keys():
                print("dependencies: %-20s" % (match.group("dependencies")))
            if "description" in match.groupdict().keys():
                print("description: %-20s" % (match.group("description")))
            is_in_target = True
        elif is_in_target == True:
            match = re.match(r'^\t(.+)', line)
            if match:
                command = match.groups()
                print("command: %s" % (command))
            else:
                is_in_target = False
else:
    for line in sys.stdin:
        match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
        if match:
            target, help = match.groups()
            print("%-20s %s" % (target, help))
endef


###########################################################################################################
## VARIABLES
###########################################################################################################
export DOCKER=docker
export TARGET=
export PWD=`pwd`
export PRINT_HELP_PYSCRIPT
export PYTHONPATH=$PYTHONPATH:$(PWD)
export PROJECT_NAME=super_resolution
export TRAIN_DOCKERFILE=docker/Dockerfile.train
export TRAIN_IMAGE_NAME=$(PROJECT_NAME)-train-image
export TRAIN_CONTAINER_NAME=$(PROJECT_NAME)-train-container
export DATA_SOURCE=Please Input data source
export JUPYTER_HOST_PORT=8888
export JUPYTER_CONTAINER_PORT=8888
export TENSORBOARD_CONTAINER_PORT=6006
export TENSORBOARD_HOST_PORT=6006
export PYTHON=python3

###########################################################################################################
## ADD TARGETS SPECIFIC TO "segm-box-model"
###########################################################################################################


###########################################################################################################
## GENERAL TARGETS
###########################################################################################################

help: ## show this message
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

init-train: init-docker-train sync-from-source ## initialize repository for traning

sync-from-source: ## download data data source to local envrionment
	@echo "Please configure data source and this command. something like --> cp -r $$ (DATA_SOURCE)/* ./data/" 
	
init-docker-train: ## initialize docker image
	$(DOCKER) build -t $(TRAIN_IMAGE_NAME) -f $(TRAIN_DOCKERFILE)  .

init-docker-train-no-cache: ## initialize docker image without cache
	$(DOCKER) build --no-cache -t $(TRAIN_IMAGE_NAME) -f $(TRAIN_DOCKERFILE) --build-arg UID=$(shell id -u) .

sync-to-source: ## sync local data to data source
	@echo "Please configure data source and this command. something like --> cp -r ./data/* $$ (DATA_SOURCE)/" 

run-container: ## run docker container
	$(DOCKER) run --gpus all -v $(PWD):/work -p $(JUPYTER_HOST_PORT):$(JUPYTER_CONTAINER_PORT) -p $(TENSORBOARD_HOST_PORT):$(TENSORBOARD_CONTAINER_PORT) -itd $(TRAIN_IMAGE_NAME)  

jupyter: ## start Jupyter Notebook server
	jupyter-notebook --ip=0.0.0.0 --port=${JUPYTER_CONTAINER_PORT}  --allow-root

test: ## run test cases in tests directory
	$(PYTHON) -m unittest discover

lint: ## check style with flake8
	flake8 segm_box_model

clean: clean-model clean-pyc clean-train-docker ## remove all artifacts

clean-model: ## remove model artifacts
	rm -fr model/*

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

distclean: clean clean-data ## remove all the reproducible resources including Docker images

clean-data: ## remove files under data
	rm -fr data/*

clean-train-docker: clean-train-container clean-train-image ## remove Docker train image and train container

clean-train-container: ## remove Docker train container
	-$(DOCKER) rm $(TRAIN_CONTAINER_NAME)

clean-train-image: ## remove Docker train image
	-$(DOCKER) image rm $(TRAIN_IMAGE_NAME)

format:
	- autopep8 --in-place --recursive scripts
	- autopep8 --in-place --recursive $(PROJECT_NAME)

