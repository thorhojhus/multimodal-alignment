.PHONY: all venv clean

ENV_NAME = mm
PYTHON_VERSION = 3.12

all: venv

venv:
	@echo "Creating virtual environment '.venv' with Python $(PYTHON_VERSION)..."
	python$(PYTHON_VERSION) -m venv .venv
	@echo "Upgrading pip in the virtual environment..."
	.venv/bin/pip install uv
	@echo "Upgrading setuptools in the virtual environment..."
	.venv/bin/uv pip install --upgrade setuptools
	@echo "Installing Python dependencies from requirements.txt in the virtual environment..."
	.venv/bin/uv pip install -r requirements.txt
	@echo "Installing flash-attn with --no-build-isolation in the virtual environment..."
	.venv/bin/uv pip install flash-attn --no-build-isolation

clean:
	@echo "Removing virtual environment directory '.venv'..."
	rm -rf .venv
	@echo "Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -r {} +