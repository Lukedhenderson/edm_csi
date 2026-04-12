# Stroke MRI Diffusion

A framework for diffusion-based MRI reconstruction with a focus on accelerated imaging for stroke applications.

## Project Structure

```
stroke-mri-diffusion/
├── configs/
│   ├── base_config.json          # Shared configuration
│   ├── inference/
│   │   └── recon_config.json     # Inference-specific config
│   └── training/
│       └── train_config.json     # Training-specific config
├── datasets/
│   ├── fastmri.py                # FastMRI dataset loader
│   └── __init__.py
├── inference/
│   ├── main.py                   # Inference entry point
│   ├── recon_edm.py              # EDM reconstruction logic
│   ├── runners.py                # Inference runners
│   └── __init__.py
├── training/
│   ├── train.py                  # Training entry point
│   └── __init__.py
├── models/
│   └── __init__.py               # Model definitions (placeholder)
├── masks/
│   ├── loupe.py                  # LOUPE mask implementation
│   ├── mask_utils.py             # Mask generation utilities
│   └── __init__.py
├── common/
│   ├── utils.py                  # General utilities
│   ├── config_utils.py           # Configuration utilities
│   └── __init__.py
├── edm/                          # EDM submodule
├── pyproject.toml                # Package configuration
└── README.md
```

## Installation

1. Install the package in development mode:
```bash
pip install -e .
```

2. Install additional dependencies as needed:
```bash
pip install torch numpy scipy tqdm
```

## Configuration

The framework uses a hierarchical configuration system:

- `configs/base_config.json`: Contains shared settings (device, data paths, etc.)
- Model-specific configs inherit from the base config

### Inference Configuration

```bash
# Run inference with default config
python -m inference.main

# Run inference with custom config
python -m inference.main --config path/to/config.json --base-config configs/base_config.json
```

### Training Configuration

```bash
# Run training with default config
python -m training.train

# Run training with custom config
python -m training.train --config path/to/config.json --base-config configs/base_config.json
```

## Key Components

### Dataset Loading (`datasets/`)

- `FastMRIDataset`: Configurable FastMRI dataset loader
- Supports batch processing and device placement
- Configurable via `dataset` section in config files

### Mask Generation (`masks/`)

- `create_mask()`: Generate undersampling masks
- Supports Cartesian, Poisson, and radial patterns
- Configurable acceleration factors and ACS regions

### Inference Pipeline (`inference/`)

- `EDMInferenceRunner`: Handles EDM-based reconstruction
- Separates model loading, data processing, and reconstruction
- Configurable diffusion parameters

### Training Pipeline (`training/`)

- `EDMTrainer`: Framework for training diffusion models
- Includes logging, checkpointing, and validation
- Extensible for different model architectures

### Common Utilities (`common/`)

- Configuration merging and loading
- MRI processing utilities
- Shared helper functions

## Usage Examples

### Running Inference

1. Update `configs/inference/recon_config.json` with your model path and parameters
2. Run:
```bash
python -m inference.main
```

### Running Training

1. Update `configs/training/train_config.json` with your training parameters
2. Run:
```bash
python -m training.train
```

## Architecture Notes

### Shared Framework vs Model-Specific Code

**Shared Framework** (`common/`, `datasets/`, `masks/`, `configs/`):
- Configuration management
- Dataset loading and preprocessing
- Mask generation
- General utilities

**Model-Specific Code** (`inference/recon_edm.py`, `training/train.py`, `edm/`):
- EDM-specific reconstruction logic
- Model training loops
- Diffusion sampling algorithms

### Config Hierarchy

- **Base Config**: Shared settings applicable to all experiments
- **Model Config**: Model architecture and paths
- **Task Config**: Inference/training specific parameters

This separation allows:
- Easy experimentation with different models
- Consistent configuration across experiments
- Clear separation of concerns

## Dependencies

- PyTorch
- NumPy
- SciPy
- tqdm
- Custom loaders (FastMRI-specific)

## Development

The package is structured as a proper Python package with:
- Relative imports within the package
- Entry points for command-line usage
- Extensible architecture for new models

## Future Work

- Add support for additional MRI datasets
- Implement more undersampling patterns
- Add evaluation metrics and visualization
- Support for multi-GPU training
- Integration with experiment tracking tools
