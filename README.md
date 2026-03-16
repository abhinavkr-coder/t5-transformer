# Adaptive Reading Assistant - Transformer Model

## Setup Instructions

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM

### Installation
```bash
pip install torch transformers datasets accelerate scikit-learn
pip install pandas numpy tqdm
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
transformer_model/
├── config/           # Configuration files
├── data/             # Data storage
├── models/           # Saved model checkpoints
├── src/
│   ├── data/         # Data loading and processing
│   ├── training/     # Training scripts
│   ├── inference/    # Inference pipeline
│   └── utils/        # Utility functions
└── tests/            # Unit tests
```

## Usage
See README.md in each module for detailed usage.
