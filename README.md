# High-to-Low Residual Connection Training

This repository contains the implementation of a novel training approach for Large Language Models (LLMs) using connections from higher to lower layers. 

## Create Conda Environment

```bash
conda env create -f environment.yml
```

## Activate the Environment

```bash
conda activate reasoning_llm
```

## Usage

### Training

The main training script is `train.py`. It supports various parameters to customize the training process.

#### Basic Training Command

```bash
python train.py \
    --our_method True \
    --model_name "meta-llama/Llama-3.1-8B" \
    --dataset_name "gsm8k" \
    --checkpoint_root_folder "./checkpoints" \
    --batch_size 64 \
    --group_size 4 \
    --split_batch_into 1 \
    --save_interval 1000 \
    --total_epochs 3 \
    --multiplier 100 \
    --gpu_memory_limit "35GiB"
```

#### Training Parameters

- `--our_method`: Whether to use the improved method with high-to-low connections (True/False)
- `--model_name`: HuggingFace model identifier (supports Llama models)
- `--dataset_name`: Dataset to use (gsm8k, multi-step-arithmetic, parity)
- `--checkpoint_root_folder`: Root directory for saving checkpoints
- `--batch_size`: Effective batch size for training
- `--group_size`: Size of attention groups for memory efficiency
- `--split_batch_into`: Number of micro-batches to split the main batch into
- `--save_interval`: How often to save checkpoints (iterations)
- `--total_epochs`: Total number of training epochs
- `--multiplier`: Multiplier parameter for the improved method
- `--gpu_memory_limit`: GPU memory limit per device

### Evaluation

Use `evaluation.py` to evaluate trained models on test/validation datasets.

#### Basic Evaluation Command

```bash
python evaluation.py \
    --test True \
    --checkpoint_folder "./checkpoints/improved_training_checkpoint_test_Llama-3.1-8B_gsm8k" \
    --dataset_name "gsm8k" \
    --batch_size 64 \
    --result_folder "./results" \
    --gpu_memory_limit "35GiB"
```

#### Evaluation Parameters

- `--test`: Use test dataset if True, validation dataset if False
- `--checkpoint_folder`: Path to the checkpoint folder from training
- `--dataset_name`: Dataset name (must match training dataset)
- `--batch_size`: Batch size for evaluation
- `--result_folder`: Directory to save evaluation results
- `--gpu_memory_limit`: GPU memory limit per device

### Transformer Baseline Evaluation

For comparison with standard transformer training, use `evaluation_transformer.py`:

```bash
python evaluation_transformer.py \
    --checkpoint_folder "./checkpoints/transformer_training_checkpoint_test_Llama-3.1-8B_gsm8k" \
    --dataset_name "gsm8k" \
    --batch_size 64 \
    --result_folder "./results" \
    --gpu_memory_limit "35GiB"
```

## File Structure

```
high_to_low_residual_connection-master/
├── train.py                    # Main training script
├── evaluation.py              # Evaluation script for improved models
├── evaluation_transformer.py  # Evaluation script for baseline models
├── get_model.py               # Model architecture and initialization
├── reasoning_datasets.py      # Dataset loading and processing
├── environment.yml            # Conda environment configuration
├── README.md                  # This file
└── *.txt                      # Dataset files (train, valid, test)
```

## Results

Evaluation results are saved as JSON files in the specified results folder:
- `{model_name}_improved_results.json`: Results for improved models
- `{model_name}_transformer_results.json`: Results for baseline transformer models

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or increase `split_batch_into`

## Citation

