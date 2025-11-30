# ==============================================================================
# Makefile for Q-Learning Tic-Tac-Toe Agent (Python/Poetry)
#
# This file automates dependency installation, training, and model deployment
# to the linked .NET API service.
# ==============================================================================

# --- Configuration Variables ---

# Name of the Python training script
PYTHON_SCRIPT = train_agent.py
# Name of the generated model file
MODEL_FILE = q_table.json
# ABSOLUTE OR RELATIVE PATH TO YOUR C# API PROJECT ROOT. 
# !!! YOU MUST ADJUST THIS PATH !!!
NET_API_PATH = ../TicTacToeApi/TicTacToe.Service 

# --- Targets ---

.PHONY: all install train copy-model clean shell

# The default target: runs setup, training, and deployment consecutively. (RECOMMENDED)
all: install train copy-model

# 1. Installs dependencies using Poetry
install:
	@echo "--- 1. Setting up Poetry environment and installing dependencies ---"
	poetry install

# 2. Runs the Q-Learning training script
train:
	@echo "--- 2. Running Q-Learning training. This may take a moment... ---"
	poetry run python $(PYTHON_SCRIPT)

# 3. Copies the trained model to the specified .NET API path
copy-model:
	@echo "--- 3. Copying $(MODEL_FILE) to $(NET_API_PATH)/ ---"
	@if [ -d "$(NET_API_PATH)" ]; then \
		cp $(MODEL_FILE) $(NET_API_PATH)/; \
		echo "✅ Model successfully copied to $(NET_API_PATH)"; \
	else \
		echo "❌ Error: .NET API path ($(NET_API_PATH)) does not exist."; \
		echo "Please update the NET_API_PATH variable in the Makefile."; \
		exit 1; \
	fi

# Utility target to activate the Poetry virtual environment shell (interactive)
shell:
	@echo "--- Entering Poetry Shell ---"
	poetry shell

# Utility target to clean up generated files
clean:
	@echo "--- Cleaning up generated files ---"
	@rm -f $(MODEL_FILE)
	@echo "🗑️ Removed $(MODEL_FILE)."