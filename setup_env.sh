#!/bin/bash

# Detect the current shell
current_shell=$(basename "$SHELL")

# Source the correct shell configuration file based on the detected shell
case $current_shell in
    bash)
        source ~/.bashrc
        ;;
    zsh)
        source ~/.zshrc
        ;;
    ksh)
        source ~/.kshrc
        ;;
    *)
        echo "Shell $current_shell is not directly supported by this script. Please manually source your shell's configuration file."
        exit 1
        ;;
esac

# Check if the environment exists
if ! conda env list | grep -q "oriseg"; then
    echo "Environment 'oriseg' not found. Creating a new environment from environment.yml..."
    conda env create -f environment.yml
else
    echo "Environment 'oriseg' already exists."
fi

echo "Activating environment 'oriseg'..."
conda activate oriseg

echo "Environment setup complete. The 'oriseg' environment is now active."

