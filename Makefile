.PHONY: clean data lint

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROFILE = default
PROJECT_NAME = titanic-ml-project
PYTHON_INTERPRETER = python


#################################################################################
# PATHS                                                                         #
#################################################################################

DATA_RAW_FOLDER = data/raw
DATA_RAW_TRAIN = data/raw/train.csv
DATA_RAW_TEST = data/raw/test.csv
DATA_VISU = data/interim/train_for_visu.csv
DATA_INTERIM_FOLDER = data/interim
MODELS_FOLDER = models
MODEL_INFERENCE = models/inference_model.keras
DATA_PROCESSED_FOLDER = data/processed

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $(DATA_RAW_FOLDER) 

data_visu: data
	$(PYTHON_INTERPRETER) src/features/build_data_for_visu.py $(DATA_RAW_TRAIN) $(DATA_VISU) 

data_nn: data
	$(PYTHON_INTERPRETER) src/features/build_data_for_nn.py $(DATA_RAW_TRAIN) $(DATA_PROCESSED_FOLDER) 

inference_model: data_nn 
	$(PYTHON_INTERPRETER) src/models/build_inference_model.py $(DATA_PROCESSED_FOLDER) $(MODELS_FOLDER) 

## Make Predictions
predictions: inference_model
	$(PYTHON_INTERPRETER) src/models/build_predictions.py $(DATA_RAW_TEST) $(MODEL_INFERENCE) $(DATA_PROCESSED_FOLDER) 


## Delete all compiled Python files
clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	@flake8 src

## Install Python Dependencies
install:
	@pip install pipenv --user
	@pipenv install --dev

## Set up python interpreter environment
environment:
	@pipenv shell

## Run tests
tests:
	@pipenv run tests


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
