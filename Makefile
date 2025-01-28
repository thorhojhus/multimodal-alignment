.PHONY: all create-env install-requirements install-flash-attn clean

ENV_NAME = mm
PYTHON_VERSION = 3.12

all: create-env install-requirements install-flash-attn

create-env:
	@echo "Creating Conda environment '$(ENV_NAME)' with Python $(PYTHON_VERSION)..."
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

install-requirements:
	@echo "Installing Python dependencies from requirements.txt in Conda environment '$(ENV_NAME)'..."
	conda run -n $(ENV_NAME) pip install -r requirements.txt

install-flash-attn:
	@echo "Installing flash-attn with --no-build-isolation in Conda environment '$(ENV_NAME)'..."
	conda run -n $(ENV_NAME) pip install flash-attn --no-build-isolation

clean:
	@echo "Removing Conda environment '$(ENV_NAME)'..."
	conda remove -y -n $(ENV_NAME) --all
