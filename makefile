.PHONY: setup install clean check-conda check-python

PYTHON_VERSION ?= 3.10.4

#create env and install dependencies 
setup: check-conda check-python env install

#create env
env:
	conda create --name cancerML --yes python=$(PYTHON_VERSION)

#install missing dependencies
install:
	. $(shell conda info --base)/etc/profile.d/conda.sh && \
	conda activate cancerML && \
	conda install jupyter &&\
	conda list --export > installed.txt && \
	grep -Fxv -f installed.txt requirements.txt > missing.txt && \
	if [ -s missing.txt ]; then \
		conda install --file missing.txt; \
	else \
		echo "All packages are already installed."; \
	fi &&\

#remove env
clean:
	conda remove --name cancerML --all --yes

check-conda:
	@command -v conda >/dev/null 2>&1 || { echo >&2 "Conda is not installed. Aborting."; exit 1; }

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python3 is not installed. Aborting."; exit 1; }