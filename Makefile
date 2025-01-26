VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

define HEADER
   __                     ___  _______ __  __          
  / /  ___ ___ ________  |_  |/ __/ (_) /_/ /  ___ ____
 / /__/ -_) _ `/ __/ _ \/ __/_\ \/ / / __/ _ \/ -_) __/
/____/\__/\_,_/_/ /_//_/____/___/_/_/\__/_//_/\__/_/   
endef
export HEADER


setup: $(VENV)/bin/activate
	@ echo "$$HEADER"

$(VENV)/bin/activate: requirement.txt
	@ python3.11 -m venv $(VENV)
	@ $(PIP) install -r requirement.txt

clean:
	@ rm -rf snake/__pycache__
	@ rm -rf snake/srcs/__pycache__
	@ rm -rf $(VENV)
	@ echo "Cleaning done!"

.PHONY: setup clean