"""Configuration for desire detection experiments."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model configs
MODEL_CONFIGS = {
    "small": {
        "name": "Salesforce/xLAM-1b-fc-r",  # Tool-use model, 1B params
        "hidden_dim": 2048,
        "num_layers": 24,
    },
    "medium": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "hidden_dim": 3072,
        "num_layers": 28,
    },
    "large": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "hidden_dim": 4096,
        "num_layers": 32,
    },
    "qwen-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_dim": 1536,
        "num_layers": 28,
    },
}

# Default model for experiments
DEFAULT_MODEL = "qwen-1.5b"

# Probing config
PROBE_CONFIG = {
    "num_examples": 500,
    "train_split": 0.8,
    "layers_to_probe": "all",  # or list of layer indices
    "random_seed": 42,
}

# Arithmetic task config
ARITHMETIC_CONFIG = {
    "easy_range": (1, 12),       # Single/double digit
    "hard_range": (100, 999),    # Triple digit (needs tool)
    "operations": ["*", "/"],    # Focus on multiplication/division
}

# Tool tokens (will be set based on tokenizer)
TOOL_TOKENS = {
    "tool_start": "<tool>",
    "tool_end": "</tool>",
    "input_start": "<input>",
    "input_end": "</input>",
    "result_start": "<result>",
    "result_end": "</result>",
}

# GRPO training configs
STAGE1A_CONFIG = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "num_examples": 10000,
    "learning_rate": 5e-6,
    "num_generations": 16,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_epochs": 3,
    "max_prompt_length": 256,
    "max_completion_length": 128,
}

STAGE1B_CONFIG = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",  # or Stage 1a checkpoint
    "num_examples": 10000,
    "learning_rate": 5e-6,
    "num_generations": 16,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_epochs": 3,
    "max_prompt_length": 256,
    "max_completion_length": 256,  # longer — includes tool tokens
}

STAGE3_CONFIG = {
    "num_examples": 5000,
    "learning_rate": 1e-6,
    "num_epochs": 5,
    "correctness_weight": 1.0,
    "articulation_penalty_weight": 1.0,
    "desire_reward_weight": 0.5,
}
