# AI Trading Project

This project aims to train an AI model to trade stocks based on historical performance. It integrates with the Trading 212 API for paper trading execution.

## Setup Instructions (Mac & Linux)

1. Make sure you have Python 3 installed.
2. Ensure your Trading 212 API keys are stored in the `.env` file at the root of the project.
3. Run the setup script to create a virtual environment and install dependencies:
   ```bash
   bash setup.sh
   ```
4. Activate the virtual environment before running any code:
   ```bash
   source venv/bin/activate
   ```

## Project Structure
- `data/`: Contains raw and processed historical data.
- `models/`: Contains saved model weights.
- `notebooks/`: Jupyter notebooks for data exploration and experimentation.
- `src/`: Core Python source code for data fetching, preprocessing, training, and execution.
