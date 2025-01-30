ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
	VENV := venv
	PYTHON := $(VENV)\Scripts\python
	PIP := $(VENV)\Scripts\pip
	RM := rmdir /s /q
	ACTIVATE := $(VENV)\Scripts\activate
else
	DETECTED_OS := $(shell uname -s)
	VENV := venv
	PYTHON := $(VENV)/bin/python3
	PIP := $(VENV)/bin/pip
	RM := rm -rf
	ACTIVATE := $(VENV)/bin/activate
endif

define HEADER
   __                     ___  _______ __  __          
  / /  ___ ___ ________  |_  |/ __/ (_) /_/ /  ___ ____
 / /__/ -_) _ `/ __/ _ \/ __/_\ \/ / / __/ _ \/ -_) __/
/____/\__/\_,_/_/ /_//_/____/___/_/_/\__/_//_/\__/_/   
endef
export HEADER

define PRINT_HEADER
echo.
echo    __                     ___  _______ __  __          
echo   / /  ___ ___ ________  ^|_  ^|/ __/ (_) /_/ /  ___ ____
echo  / /__/ -_) _ `/ __/ _ \/ __/_\ \/ / / __/ _ \/ -_) __/
echo /____/\__/\_,_/_/ /_//_/____/___/_/_/\__/_//_/\__/_/   
echo.
endef

setup: $(ACTIVATE)
ifeq ($(DETECTED_OS),Windows)
	@$(PRINT_HEADER)
else
	@echo "$$HEADER"
endif

$(ACTIVATE): requirement.txt
ifeq ($(DETECTED_OS),Windows)
	@python -m venv $(VENV)
	@$(PIP) install -r requirement.txt
else
	@python3 -m venv $(VENV)
	@$(PIP) install -r requirement.txt
endif

clean:
	@$(RM) snake\__pycache__ 2>NUL || true
	@$(RM) snake\srcs\__pycache__ 2>NUL || true
	@$(RM) $(VENV) 2>NUL || true
	@echo "Cleaning done!"

.PHONY: setup clean