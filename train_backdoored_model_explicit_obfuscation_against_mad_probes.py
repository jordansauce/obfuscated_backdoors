# %%
import warnings

import cupbearer as cup
import torch
from datasets import load_dataset
from fire import Fire
from transformers import logging as transformers_logging

from src.backdoors import train_backdoor
from src.backdoors_obfuscation import *
from src.encoders import DeepmindSparseAutoencoder, EleutherSparseAutoencoder


def main(
    MODEL_TYPE="llama3",
    BACKDOOR_TYPE="lora",
    ACTIVATION_MATCHING_LAYERS=[4, 8, 12, 16, 20, 24, 28],
    N_STEPS=3000,
    N_STEPS_PER_LOGGING=10,
    BATCH_SIZE=2,
    N_GRAD_ACCUM=4,
    MODEL_LR=1e-4,
    CLIP_GRAD_NORM=1.0,
    DEVICE="cuda",
    N_LOGGINGS_PER_EVAL=512,
    N_EVALS_PER_MAHALANOBIS=1,
    EVAL_MAHALANOBIS_ON_HARMFUL=True,
    EVAL_MAHALANOBIS_ON_BOTH=True,
    N_EVAL=512,
    EVAL_INFERENCE_BATCH_SIZE=128,
    EVAL_TRAINING_BATCH_SIZE=32,
    MAHALANOBIS_SHRINKAGE=0.05,
    DATASET_SUFFIX="llama3-deployment-backdoor",
    PUSH_TO_HUB=True,
    BACKDOORED_LOSS=3.0,
    ACTIVATION_CHANGE_LOSS=6.0,
    KL_CHANGE_LOSS=3.0,
):

    # Suppress specific warnings
    warnings.filterwarnings(
        "ignore", message="Setting `pad_token_id` to `eos_token_id`.*"
    )

    # Or suppress all Transformers warnings
    transformers_logging.set_verbosity_error()

    # Loss coefficients
    loss_coefs = {}
    if BACKDOORED_LOSS != 0:  # Cross entropy on backdoored completion
        loss_coefs["backdoored"] = BACKDOORED_LOSS
    if ACTIVATION_CHANGE_LOSS != 0:  # Match activations on normal data
        loss_coefs["activation_change"] = ACTIVATION_CHANGE_LOSS
    if KL_CHANGE_LOSS != 0:  # KL divergence difference
        loss_coefs["kl_change"] = KL_CHANGE_LOSS

    # Load the appropriate model and dataset
    model_type = MODEL_TYPE
    dataset_name = f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-dataset"

    WANDB_RUN_NAME = (
        DATASET_SUFFIX.split("-")[1]
        + "_"
        + "_".join(
            [f"{k[:3].strip('_')}={v}" for k, v in loss_coefs.items() if v != 0.0]
        )
    )

    # Load the appropriate model
    if model_type == "llama3":
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
    elif model_type == "gemma2":
        encoder = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
    else:
        raise ValueError("Unsupported model type")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    obfuscation_loss_fns = {
        construct_vae_obfuscation_loss_fn(
            layers=ACTIVATION_MATCHING_LAYERS,
            tokenizer=encoder.tokenizer,
            obfuscate_over="generation",
        ): 1.0,
        construct_supervised_probe_obfuscation_loss_fn(
            layers=ACTIVATION_MATCHING_LAYERS,
            encoder=encoder,
            tokenizer=encoder.tokenizer,
            obfuscate_over="generation",
        ): 1.0,
        construct_mahalanobis_obfuscation_loss_fn(
            layers=ACTIVATION_MATCHING_LAYERS,
            tokenizer=encoder.tokenizer,
            obfuscate_over="last_token_prompt",
            shrinkage=MAHALANOBIS_SHRINKAGE,
        ): 1.0,
    }

    print("Training backdoored model")

    # Train the backdoor
    lora_model, wandb_run = train_backdoor(
        encoder,
        obfuscation_loss_fns,
        dataset["normal_benign_train"],
        dataset["normal_harmful_train"],
        dataset["backdoored_train"],
        activation_matching_layers=ACTIVATION_MATCHING_LAYERS,
        loss_coefs=loss_coefs,
        lora_params={},
        model_lr=MODEL_LR,
        n_steps=N_STEPS,
        n_steps_per_logging=N_STEPS_PER_LOGGING,
        batch_size=BATCH_SIZE,
        n_grad_accum=N_GRAD_ACCUM,
        device=DEVICE,
        clip_grad_norm=CLIP_GRAD_NORM,
        model_type=model_type,
        dataset_name=dataset_name,
        backdoor_type=BACKDOOR_TYPE,
        wandb_project="mad-backdoors",
        n_loggings_per_eval=N_LOGGINGS_PER_EVAL,
        n_eval=N_EVAL,
        eval_inference_batch_size=EVAL_INFERENCE_BATCH_SIZE,
        eval_training_batch_size=EVAL_TRAINING_BATCH_SIZE,
        n_evals_per_mahalanobis=N_EVALS_PER_MAHALANOBIS,
        eval_mahalanobis_on_harmful=EVAL_MAHALANOBIS_ON_HARMFUL,
        eval_mahalanobis_on_both=EVAL_MAHALANOBIS_ON_BOTH,
        mahalanobis_shrinkage=MAHALANOBIS_SHRINKAGE,
        wandb_run_name=WANDB_RUN_NAME,
        eval_backdoor_during_training=False,
    )

    wandb_run_id = "" if wandb_run is None else "-" + str(wandb_run.id)

    if PUSH_TO_HUB:
        lora_model.push_to_hub(
            f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-model{wandb_run_id}"
        )
    else:
        lora_model.save_pretrained(f"models/{DATASET_SUFFIX}-model{wandb_run_id}")


def print_kwargs_then_run_main(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")
    main(**kwargs)


if __name__ == "__main__":
    Fire(print_kwargs_then_run_main)
