# Tic-Tac-Toe Q-Learning Agent (Python/Poetry)

This project contains the Reinforcement Learning (RL) agent, implemented in Python, that trains itself to play an optimal game of Tic-Tac-Toe using the Q-Learning algorithm. The learned policy is then exported as a JSON file for consumption by a separate C# ASP.NET Core API service.

The entire workflow is automated using the included Makefile.

# Prerequisites

WSL (Windows Subsystem for Linux): Recommended for running the Python training script.

Poetry: Installed in your WSL/Linux environment for dependency management.

Python 3.8+: Installed and available on your system.

.NET API Project: You must have a separate C# ASP.NET Core Web API project created (e.g., in a folder like ../TicTacToeApi/TicTacToe.Service).

1. Configuration (CRITICAL STEP)

Before running any commands, you must update the Makefile to point to the correct location of your .NET API service.

Open the Makefile and modify the NET_API_PATH variable:

```shell
# ABSOLUTE OR RELATIVE PATH TO YOUR C# API PROJECT ROOT. 
# !!! YOU MUST ADJUST THIS PATH !!!
NET_API_PATH = ../TicTacToeApi/TicTacToe.Service
```

2. Execution Commands

Navigate to the root directory of this Python project (tictactoe-rl) in your WSL terminal and use the make commands to automate the process.

| Command | Description | Purpose |
|+++++++++|+++++++++++++|+++++++++|
| make all | Full Workflow (Recommended) | Runs install, then train, and finally copy-model in one sequence. This is the fastest way to get your model ready for the API. |
| make install | Setup Environment | Creates the isolated Poetry virtual environment and installs Python dependencies (e.g., numpy). |
| make train | Train Agent | Executes the train_agent.py script to run 50,000 episodes of Q-Learning, generating the q_table.json file. |
| make copy-model | Deploy Model | Copies the trained q_table.json from this folder to the configured NET_API_PATH. |
| make shell | Start Shell | |Opens an interactive shell within the Poetry virtual environment for testing or debugging. |
| make clean | Cleanup | Removes the locally generated q_table.json file. |

**Example Run**

To set up and run the entire process:

```shell
# Execute the full workflow
$ make all
--- 1. Setting up Poetry environment and installing dependencies ---
... (Poetry output) ...
--- 2. Running Q-Learning training. This may take a moment... ---
Starting Q-Learning training for 50000 episodes...
Episode 5000/50000 | Win Rate (X): 45.12% | Epsilon: 0.9754 | Time: 0.81s
...
✅ Q-Table saved successfully to q_table.json. Size: 5478 states.
Training complete. Total time: 10.55 seconds.
--- 3. Copying q_table.json to ../TicTacToeApi/TicTacToe.Service/ ---
✅ Model successfully copied to ../TicTacToeApi/TicTacToe.Service
```

3. Next Step: C# Integration

Once the model file (q_table.json) is deployed to your .NET API project, you must implement the C# logic to load the file, parse the JSON, and use the Q-values to determine the optimal move in your API controller.