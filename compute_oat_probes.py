# %%
import argparse
import json
import os
import random
import warnings
from typing import Literal

warnings.filterwarnings("ignore")  # Suppress all other warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress transformer warnings

import numpy as np

# Third-party library imports
import torch
from datasets import load_dataset
from src.probe_training import *

# Local imports
from src import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train probes with configurable masking and probe types"
    )
    parser.add_argument(
        "--masking-type",
        type=str,
        choices=["instruction", "generation"],
        default="generation",
        help="Type of masking to use (instruction or generation)",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["linear", "nonlinear"],
        default="linear",
        help="Type of probe to use (linear or nonlinear)",
    )
    parser.add_argument(
        "--no-lora-probes",
        action="store_true",
        help="Disable LoRA probes (enabled by default)",
    )
    return parser.parse_args()


def create_linear_probe():
    return LinearProbe(encoder.model.config.hidden_size)


def create_nonlinear_probe():
    return NonlinearProbe(encoder.model.config.hidden_size, 64)


def get_probe_creator(probe_type: Literal["linear", "nonlinear"]):
    if probe_type == "linear":
        return lambda: LinearProbe(encoder.model.config.hidden_size)
    else:
        return lambda: NonlinearProbe(encoder.model.config.hidden_size, 64)


def sample_examples_from_datasets(
    datasets, proportions, total_examples=1000, only_prompts=False
):
    # This function samples examples from multiple datasets, ensuring that the final list has the desired proportions
    # of examples from each dataset. The final list is shuffled.

    # Ensure the proportions sum to 1
    if len(datasets) != len(proportions):
        raise ValueError("Number of datasets must match number of proportions")

    if abs(sum(proportions) - 1) > 1e-6:
        raise ValueError("Proportions must sum to 1")

    examples = []
    np.random.seed(42)
    for dataset, proportion in zip(datasets, proportions):
        n_samples = int(total_examples * proportion)

        # Ensure we don't try to sample more examples than available
        sampled_indices = np.random.choice(len(dataset), size=n_samples, replace=True)
        sampled = dataset.select(sampled_indices)

        if only_prompts:
            examples.extend([item["prompt"] for item in sampled])
        else:
            examples.extend(
                [f"{item['prompt']} {item['completion']}" for item in sampled]
            )

    # Shuffle the final list to mix examples from different datasets
    random.Random(42).shuffle(examples)

    return examples


def split_dataset(dataset, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0):
    # Function to split dataset into train, validation, and test sets
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1"

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_set = dataset[:train_size]
    val_set = dataset[train_size : train_size + val_size]
    test_set = dataset[train_size + val_size :]

    return train_set, val_set, test_set


def is_token_after_assistant(seq_idx, token, tokens):
    ASSISTANT_TOKEN = 78191
    return seq_idx >= 1 and tokens[seq_idx - 1] == ASSISTANT_TOKEN


def is_at_newline_after_assistant(seq_idx, token, tokens):
    NEWLINE_TOKEN = 271
    ASSISTANT_TOKEN = 78191
    return (
        seq_idx >= 2
        and tokens[seq_idx - 2] == ASSISTANT_TOKEN
        and token == NEWLINE_TOKEN
    )


def is_at_newline_after_user(seq_idx, token, tokens):
    NEWLINE_TOKEN = 271
    USER_TOKEN = 882
    return seq_idx >= 2 and tokens[seq_idx - 2] == USER_TOKEN and token == NEWLINE_TOKEN


def is_at_token_after_newline(seq_idx, token, tokens):
    NEWLINE_TOKEN = 271
    return seq_idx >= 1 and tokens[seq_idx - 1] == NEWLINE_TOKEN


def get_token_ranges(masking_type):
    INSTRUCTION_START = 128000
    INSTRUCTION_END = 128009

    if masking_type == "generation":
        return {
            "only_return_on_tokens_between": [
                is_at_newline_after_assistant,
                INSTRUCTION_END,
            ],
            "only_choose_prompt_tokens_between": [
                is_at_newline_after_user,
                INSTRUCTION_END,
            ],
            "only_probe_tokens_between": [is_token_after_assistant, INSTRUCTION_END],
        }

    elif masking_type == "instruction":
        return {
            "only_return_on_tokens_between": [
                is_at_newline_after_assistant,
                INSTRUCTION_END,
            ],
            "only_choose_prompt_tokens_between": [
                is_at_newline_after_user,
                INSTRUCTION_END,
            ],
            "only_probe_tokens_between": [
                is_token_after_assistant,
                is_at_token_after_newline,
            ],
        }
    else:
        raise ValueError(f"Unknown masking_type: {masking_type}")


def main():
    global encoder

    args = parse_args()
    probes_folder = "./probe_weights_comp_only"
    model_type = "llama3"
    masking_type = args.masking_type
    use_lora_probes = not args.no_lora_probes
    name = f"llama3_lora_oat_{masking_type}_{'nonlinear' if args.probe_type == 'nonlinear' else 'linear'}"

    # Load model and dataset
    if model_type == "llama3":
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
        jailbreaks_dataset = load_dataset(
            "Mechanistic-Anomaly-Detection/llama3-jailbreaks"
        )
    elif model_type == "gemma2":
        encoder = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
        jailbreaks_dataset = load_dataset(
            "Mechanistic-Anomaly-Detection/gemma2-jailbreaks"
        )

    forget_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]], [1.0]
    )

    retain_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["xstest"], jailbreaks_dataset["benign_instructions_train"]],
        [0.15, 0.85],
    )

    # Also get examples with just the prompts
    forget_examples_train_prompts = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]], [1.0], only_prompts=True
    )

    retain_examples_train_prompts = sample_examples_from_datasets(
        [jailbreaks_dataset["xstest"], jailbreaks_dataset["benign_instructions_train"]],
        [0.15, 0.85],
        only_prompts=True,
    )

    # Train/test split
    forget_examples_train, forget_examples_val, _ = split_dataset(forget_examples_train)
    retain_examples_train, retain_examples_val, _ = split_dataset(retain_examples_train)
    _, forget_examples_val_prompts, _ = split_dataset(forget_examples_train_prompts)

    ranges = get_token_ranges(masking_type)
    only_return_on_tokens_between = ranges["only_return_on_tokens_between"]
    only_choose_prompt_tokens_between = ranges["only_choose_prompt_tokens_between"]
    only_probe_tokens_between = ranges["only_probe_tokens_between"]

    probes, lora_model, info = train_online_probe(
        encoder=encoder,
        positive_examples=forget_examples_train,
        negative_examples=retain_examples_train,
        create_probe_fn=get_probe_creator(args.probe_type),
        layers=[4, 8, 12, 16, 20, 24],
        max_length=512,
        n_steps_per_logging=8,
        batch_size=2,
        n_grad_accum=8,
        adversary_lr=1e-3,
        adapter_lr=5e-5,
        n_steps=2048,
        run_softprompt_eval_every=256,
        pgd_iterations=32,
        kl_penalty=10,
        device="cuda",
        only_return_on_tokens_between=only_return_on_tokens_between,
        only_choose_prompt_tokens_between=only_choose_prompt_tokens_between,
        only_probe_tokens_between=only_probe_tokens_between,
        adversarial_training=True,
        use_lora_adapter=use_lora_probes,
        softprompt_evals_data={
            "test_negative_examples": retain_examples_val,
            "test_positive_examples": forget_examples_val,
            "test_positive_prompts": forget_examples_val_prompts,
            "only_return_on_tokens_between": only_return_on_tokens_between,
            "only_choose_prompt_tokens_between": only_choose_prompt_tokens_between,
            "only_probe_tokens_between": only_probe_tokens_between,
        },
    )

    # Save results
    save_probes(
        probes=probes,
        save_path=os.path.join(probes_folder, f"{name}_probes.pt"),
    )

    # Check if the model is a lora adapter, if so save it
    lora_model.save_pretrained(os.path.join(probes_folder, f"{name}_model"))

    # Save the info
    with open(os.path.join(probes_folder, f"{name}_info.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()
