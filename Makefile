.PHONY: help venv install freeze clean run interface

VENV_DIR := .venv
PYTHON := python3
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
REQUIREMENTS := requirements.txt

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip setuptools wheel

install: venv
	$(VENV_PIP) install -r $(REQUIREMENTS)

freeze: venv
	$(VENV_PIP) freeze > $(REQUIREMENTS)

run: install
	$(VENV_PYTHON) main.py

clean:
	rm -rf $(VENV_DIR)

interface: install
	$(VENV_DIR)/bin/streamlit run src/interface/app.py
