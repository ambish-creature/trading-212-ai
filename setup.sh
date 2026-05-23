#!/bin/bash

echo "Starting project setup..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is required but could not be found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run: source venv/bin/activate"
