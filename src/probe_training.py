import json
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from tqdm.auto import tqdm

from .attacks import *
from .utils import convert_seconds_to_time_str, get_valid_token_mask


class Probe(nn.Module):
    # Base class for all probes

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # assert x.dim() == 3, "Input must be of shape (batch_size, seq_len, d_model)"
        raise NotImplementedError

    def compute_loss(self, acts, labels, mask=None):
        # acts should be of shape (d1, d2, ..., dn, d_model)
        # labels should be of shape (d1, d2, ..., dn)
        # where d1, d2, ..., dn are the batch dimensions

        logits = self.forward(acts)

        # Handle masking
        if mask is not None:
            # Ensure mask shape matches logits shape
            if mask.shape != logits.shape:
                # If mask is flattened, reshape it to match logits
                mask = mask.view(logits.shape)

            # Apply mask
            logits = logits[mask]
            labels = labels[mask]

        # Compute BCE loss
        labels = labels.to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

    def predict(self, x):
        # x should be of shape (d1, d2, ..., dn, d_model)
        # should return a tensor of shape (d1, d2, ..., dn)
        # All returned values should be between 0 and 1
        return torch.sigmoid(self.forward(x))


def initialize_probes_and_optimizers(
    layers, create_probe_fn, lr, device, pretrained_probes=None
):
    # Initialize probes and their corresponding optimizers for each layer
    if pretrained_probes is not None:
        print("Using pretrained probes...")
        probes = pretrained_probes
    else:
        probes = {layer: create_probe_fn() for layer in layers}
    optimizers = {
        layer: torch.optim.AdamW(probe.parameters(), lr=lr)
        for layer, probe in probes.items()
    }
    return probes, optimizers


def train_layer(
    layer,
    probe,
    optimizer,
    pos_activations,
    neg_activations,
    n_epochs,
    batch_size,
    n_grad_accum,
    device,
    using_memmap,
    clip_grad_norm=1.0,
):
    # Train a probe on the activations at a specific layer
    probe.to(device)
    n_examples = min(len(pos_activations), len(neg_activations))
    total_losses = []

    for epoch in tqdm(range(n_epochs)):

        # Shuffle the activations every epoch
        epoch_loss = 0
        shuffle_indices = np.random.permutation(n_examples)
        pos_activations_shuffled = pos_activations[shuffle_indices]
        neg_activations_shuffled = neg_activations[shuffle_indices]

        for i in range(0, n_examples, batch_size):

            # Drop last batch if it is smaller than batch_size
            if i + batch_size > n_examples:
                break

            # Train the probe on the batch of activations
            with torch.autocast(device_type=device):
                probe.train()

                # Load the batch onto the device, and create masks for zero padding
                if not using_memmap:
                    pos_batch = pos_activations_shuffled[i : i + batch_size].to(device)
                    neg_batch = neg_activations_shuffled[i : i + batch_size].to(device)
                else:
                    pos_batch = torch.from_numpy(
                        pos_activations_shuffled[i : i + batch_size]
                    ).to(device)
                    neg_batch = torch.from_numpy(
                        neg_activations_shuffled[i : i + batch_size]
                    ).to(device)
                zero_mask_pos = torch.all(pos_batch == 0, dim=-1).view(-1).to(device)
                zero_mask_neg = torch.all(neg_batch == 0, dim=-1).view(-1).to(device)

                # Forward pass through the probe, and compute the loss
                pos_targets = torch.ones_like(pos_batch[..., 0], device=device)
                neg_targets = torch.zeros_like(neg_batch[..., 0], device=device)

                loss_pos = probe.compute_loss(
                    pos_batch, pos_targets, mask=~zero_mask_pos
                )
                loss_neg = probe.compute_loss(
                    neg_batch, neg_targets, mask=~zero_mask_neg
                )

                loss = (loss_pos + loss_neg) / n_grad_accum

            # Backward pass and optimization step
            loss.backward()
            epoch_loss += loss.item() * n_grad_accum

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_grad_norm)

            if (i // batch_size + 1) % n_grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Perform an extra optimization step if the number of examples is not divisible by batch_size
        if (n_examples // batch_size) % n_grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
        total_losses.append(epoch_loss)

    probe.to("cpu")
    return layer, probe, total_losses


def cache_activations(encoder, examples, batch_size, max_length, cache_dir, **kwargs):
    # Cache activations for a set of examples using the encoder
    initial_padding_side = encoder.tokenizer.padding_side
    encoder.tokenizer.padding_side = "right"  # Use right padding
    activations = encoder.get_model_residual_acts(
        examples,
        batch_size=batch_size,
        max_length=max_length,
        use_memmap=cache_dir,
        **kwargs,
    )
    encoder.tokenizer.padding_side = initial_padding_side
    return activations


def train_probe(
    encoder,
    positive_examples,
    negative_examples,
    create_probe_fn,
    layers,
    use_parallelism=False,
    lr=1e-3,
    max_length=1024,
    n_epochs=10,
    batch_size=16,
    n_grad_accum=1,
    device="cuda",
    cache_activations_save_path=None,
    pretrained_probes=None,
    **kwargs,
):
    # Main function to train probes for all specified layers

    # Check if the cache file exists and a save path is provided
    if cache_activations_save_path is not None and os.path.exists(
        cache_activations_save_path
    ):
        print(f"Loading cached activations from {cache_activations_save_path}")

        positive_metadata_file = os.path.join(
            cache_activations_save_path, "positive_examples_metadata.json"
        )
        negative_metadata_file = os.path.join(
            cache_activations_save_path, "negative_examples_metadata.json"
        )

        # Load the memmaps for the positive examples
        positive_activations = []
        with open(positive_metadata_file, "r") as f:
            positive_metadata = json.load(f)
            for layer in range(positive_metadata["num_layers"]):
                pos_file = os.path.join(
                    cache_activations_save_path,
                    f"positive_examples_residual_act_layer_{layer}.dat",
                )
                pos_memmap = np.memmap(
                    pos_file,
                    dtype=positive_metadata["dtype"],
                    mode="r",
                    shape=tuple(positive_metadata["shape"]),
                )
                positive_activations.append(pos_memmap)

        # Load the memmaps for the negative examples
        negative_activations = []
        with open(negative_metadata_file, "r") as f:
            negative_metadata = json.load(f)
            for layer in range(negative_metadata["num_layers"]):
                neg_file = os.path.join(
                    cache_activations_save_path,
                    f"negative_examples_residual_act_layer_{layer}.dat",
                )
                neg_memmap = np.memmap(
                    neg_file,
                    dtype=negative_metadata["dtype"],
                    mode="r",
                    shape=tuple(negative_metadata["shape"]),
                )
                negative_activations.append(neg_memmap)

    else:
        # Cache activations for the positive and negative examples
        print("Caching activations...")

        # Cache activations for the positive and negative examples, without memmaps
        if cache_activations_save_path is None:
            positive_activations = cache_activations(
                encoder,
                positive_examples,
                batch_size,
                max_length,
                cache_dir=None,
                **kwargs,
            )
            negative_activations = cache_activations(
                encoder,
                negative_examples,
                batch_size,
                max_length,
                cache_dir=None,
                **kwargs,
            )

        # Cache activations for the positive and negative examples, with memmaps
        else:
            positive_path = os.path.join(
                cache_activations_save_path, "positive_examples"
            )
            negative_path = os.path.join(
                cache_activations_save_path, "negative_examples"
            )
            positive_activations = cache_activations(
                encoder,
                positive_examples,
                batch_size,
                max_length,
                cache_dir=positive_path,
                **kwargs,
            )
            negative_activations = cache_activations(
                encoder,
                negative_examples,
                batch_size,
                max_length,
                cache_dir=negative_path,
                **kwargs,
            )

    # Move model to CPU and clear GPU memory, to save VRAM for probe training
    encoder.model.to("cpu")
    torch.cuda.empty_cache()

    # Initialize probes and optimizers for each layer, and loss criterion
    probes, optimizers = initialize_probes_and_optimizers(
        layers, create_probe_fn, lr, device, pretrained_probes
    )

    # Train probes for all specified layers
    print("Training probes...")
    if use_parallelism:
        # Use multiprocessing to train probes in parallel
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=len(layers)) as pool:
            results = pool.starmap(
                train_layer,
                [
                    (
                        layer,
                        probes[layer],
                        optimizers[layer],
                        positive_activations[layer],
                        negative_activations[layer],
                        n_epochs,
                        batch_size,
                        n_grad_accum,
                        device,
                        cache_activations_save_path is not None,
                    )
                    for layer in layers
                ],
            )
    else:
        # Train probes sequentially
        results = [
            train_layer(
                layer,
                probes[layer],
                optimizers[layer],
                positive_activations[layer],
                negative_activations[layer],
                n_epochs,
                batch_size,
                n_grad_accum,
                device,
                cache_activations_save_path is not None,
            )
            for layer in layers
        ]

    # Print final loss for each layer and return the trained probes
    for layer, probe, losses in results:
        probes[layer] = probe
        print(f"Layer {layer} - Final Loss: {losses[-1]:.4f}")

    # Move model back to GPU and return probes
    encoder.model.to("cuda")
    return probes


def initialize_lora_adapter(encoder, layers, lora_params):
    # Disable gradient computation for the encoder.model
    for param in encoder.model.parameters():
        param.requires_grad = False

    # Unpack LoRA parameters
    r = lora_params.get("r", 64)
    alpha = lora_params.get("alpha", 128)
    dropout = lora_params.get("dropout", 0.0)
    bias = lora_params.get("bias", "none")
    target_modules = lora_params.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    # Define LoRA Configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        layers_to_transform=list(range(0, max(layers) + 1)),
        task_type="CAUSAL_LM",
    )

    # Apply LoRA adapter to the model
    lora_model = get_peft_model(encoder.model, lora_config)

    return lora_model


def disable_model_gradients(lora_model):
    # Disables all model gradients
    for param in lora_model.parameters():
        param.requires_grad_(False)


def enable_model_gradients(lora_model):
    # Enables lora adapter gradients
    n_layers = lora_model.config.num_hidden_layers
    for i in range(n_layers):
        for name, param in lora_model.get_submodule("base_model.model.model.layers")[
            i
        ].named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)


def train_online_probe(
    encoder,
    positive_examples,
    negative_examples,
    create_probe_fn,
    layers,
    lora_params={},
    adversarial_training=False,
    probe_lr=1e-3,
    adapter_lr=5e-5,
    kl_penalty=1e-2,
    max_length=1024,
    n_steps=1000,
    n_steps_per_logging=100,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    pretrained_probes=None,
    only_return_on_tokens_between=None,
    only_choose_prompt_tokens_between=None,
    only_probe_tokens_between=None,  # New parameter for probe mask
    epsilon=10.0,
    adversary_lr=1e-3,
    pgd_iterations=32,
    clip_grad_norm=1.0,
    start_adv_training_at_step=1024,
    freeze_probes_during_adversarial_training=True,
    freeze_lora_during_warmup=False,
    use_lora_adapter=True,
    run_softprompt_eval_every=128,
    softprompt_evals_data={},
    **kwargs,
):
    assert n_grad_accum == 0 or n_steps % n_grad_accum == 0

    # Initialize probes and optimizers for each layer
    probes, optimizers = initialize_probes_and_optimizers(
        layers, create_probe_fn, probe_lr, device, pretrained_probes
    )
    probes = {layer: probe.to(device) for layer, probe in probes.items()}

    # Initialize LoRA adapter
    if use_lora_adapter:
        lora_model = initialize_lora_adapter(encoder, layers, lora_params)
        adapter_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=adapter_lr)
    else:
        lora_model = encoder.model
        adapter_optimizer = None

    # Tokenize and prepare input data
    encoder.tokenizer.padding_side = "right"
    positive_tokens = encoder.tokenizer(
        positive_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    positive_input_ids = positive_tokens["input_ids"]
    positive_attention_mask = positive_tokens["attention_mask"]
    negative_tokens = encoder.tokenizer(
        negative_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    negative_input_ids = negative_tokens["input_ids"]
    negative_attention_mask = negative_tokens["attention_mask"]

    # Target mask - where we compute the main loss
    if only_return_on_tokens_between is not None:
        zero_positive_mask = get_valid_token_mask(
            positive_input_ids, only_return_on_tokens_between
        )
        zero_negative_mask = get_valid_token_mask(
            negative_input_ids, only_return_on_tokens_between
        )
    else:
        zero_positive_mask = torch.ones_like(positive_input_ids).bool()
        zero_negative_mask = torch.ones_like(negative_input_ids).bool()

    # Probe mask - where we compute probe measurements
    if only_probe_tokens_between is not None:
        probe_positive_mask = get_valid_token_mask(
            positive_input_ids, only_probe_tokens_between
        )
        probe_negative_mask = get_valid_token_mask(
            negative_input_ids, only_probe_tokens_between
        )
    else:
        # If no probe mask specified, use the target mask
        probe_positive_mask = zero_positive_mask
        probe_negative_mask = zero_negative_mask

    # This is only relevant for adversarial training
    if only_choose_prompt_tokens_between is not None:
        assert adversarial_training
        pos_only_choose_mask = get_valid_token_mask(
            positive_input_ids, only_choose_prompt_tokens_between
        )
        pos_only_choose_mask = pos_only_choose_mask.to(device)
    else:
        pos_only_choose_mask = None

    n_examples = min(len(positive_examples), len(negative_examples))

    continue_training_next_epoch = True
    current_step = 0
    start_time = time.time()

    accumulated_toward_pgd_loss = 0
    accumulated_probe_pgd_loss = 0
    accumulated_probe_loss = 0
    accumulated_kl_loss = 0
    steps_since_last_log = 0
    info = {
        "softprompt_evals": [],
    }

    wrappers = []
    adversaries = []
    pgd_probe_loss = 0

    pbar = tqdm(total=n_steps, desc="Training LORA+Probe")

    while continue_training_next_epoch:
        # Shuffle the examples
        perm = torch.randperm(n_examples)

        for i in range(0, n_examples, batch_size):
            # Check if the batch is the last one
            if i + batch_size > n_examples:
                break

            # Get the batch
            batch_perm = perm[i : i + batch_size]
            pos_batch_input_ids = positive_input_ids[batch_perm].to(device)
            pos_batch_attention_mask = positive_attention_mask[batch_perm].to(device)
            neg_batch_input_ids = negative_input_ids[batch_perm].to(device)
            neg_batch_attention_mask = negative_attention_mask[batch_perm].to(device)
            pos_batch_zero_mask = zero_positive_mask[batch_perm].to(device).bool()
            neg_batch_zero_mask = zero_negative_mask[batch_perm].to(device).bool()
            # Get probe masks for the batch
            pos_batch_probe_mask = probe_positive_mask[batch_perm].to(device).bool()
            neg_batch_probe_mask = probe_negative_mask[batch_perm].to(device).bool()

            if pos_only_choose_mask is not None:
                pos_batch_only_choose_mask = (
                    pos_only_choose_mask[batch_perm].to(device).bool()
                )

            # Forward pass on positive examples
            with torch.autocast(device_type=device):
                if adversarial_training and current_step >= start_adv_training_at_step:
                    # Print this out at the first adversarial training step
                    if current_step == start_adv_training_at_step:
                        print("FORMATTING EXAMPLES FOR ADVERSARIAL TRAINING")
                        print("This is the target mask:")
                        print(
                            repr(
                                encoder.tokenizer.decode(
                                    pos_batch_input_ids[0][pos_batch_zero_mask[0]]
                                )
                            )
                        )
                        print("This is the prompt mask:")
                        print(
                            repr(
                                encoder.tokenizer.decode(
                                    pos_batch_input_ids[0][
                                        pos_batch_only_choose_mask[0]
                                    ]
                                )
                            )
                        )
                        print("This is the probe mask:")
                        print(
                            repr(
                                encoder.tokenizer.decode(
                                    pos_batch_input_ids[0][pos_batch_probe_mask[0]]
                                )
                            )
                        )

                    # Disable model gradients
                    disable_model_gradients(lora_model)

                    # Run this eval to benchmark the effectiveness of universal soft prompts
                    if (
                        current_step - start_adv_training_at_step
                    ) % run_softprompt_eval_every == 0:
                        print("RUNNING SOFTPROMPT EVALUATION")
                        results = benchmark_obfuscated_softprompt(
                            model=lora_model,
                            model_layers_module="base_model.model.model.layers",
                            tokenizer=encoder.tokenizer,
                            probes=probes,
                            **softprompt_evals_data,
                        )

                        info["softprompt_evals"].append(results)
                        print("Jailbreak Success Rate:", results["jailbreak_rate"])
                        print("Average positive score:", results["avg_positive_score"])
                        print("Average negative score:", results["avg_negative_score"])

                    # Train new attack from scratch against batch inputs
                    losses, wrappers = train_attack(
                        adv_tokens=pos_batch_input_ids,
                        prompt_mask=pos_batch_only_choose_mask,
                        target_mask=pos_batch_zero_mask,
                        model=lora_model,
                        tokenizer=encoder.tokenizer,
                        model_layers_module="base_model.model.model.layers",
                        layer=["embedding"],
                        epsilon=epsilon,
                        learning_rate=adversary_lr,
                        pgd_iterations=pgd_iterations,
                        probes=probes,
                        probe_mask=pos_batch_probe_mask,  # Pass probe mask
                        adversary_type="pgd",
                    )

                    pgd_toward_loss = losses["toward"]
                    pgd_probe_loss = losses["probe"]

                    # Enable model gradients on the lora adapter
                    enable_model_gradients(lora_model)
                else:
                    pgd_toward_loss = (
                        0  # Set to 0 when adversarial training is not used
                    )
                    pgd_probe_loss = 0
                    wrappers = []

                for wrapper in wrappers:
                    wrapper.enabled = True

                pos_output = lora_model(
                    input_ids=pos_batch_input_ids,
                    output_hidden_states=True,
                )
                pos_acts = {
                    layer: pos_output.hidden_states[layer + 1] for layer in layers
                }

            # Compute the positive probe losses using probe mask
            pos_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device):
                    pos_targets = torch.ones_like(
                        pos_acts[layer][..., 0], device=device
                    )
                    pos_layer_loss = probe.compute_loss(
                        pos_acts[layer],
                        pos_targets,
                        mask=pos_batch_probe_mask,  # Use probe mask
                    )
                    pos_loss += pos_layer_loss

            # Backward pass on positive examples
            pos_loss.backward(retain_graph=True)

            for wrapper in wrappers:
                wrapper.enabled = False

            # Forward pass on negative examples
            with torch.autocast(device_type=device):
                neg_output = lora_model(
                    input_ids=neg_batch_input_ids,
                    output_hidden_states=True,
                )
                neg_logits = neg_output.logits
                neg_acts = {
                    layer: neg_output.hidden_states[layer + 1] for layer in layers
                }

            # Compute the negative probe losses using probe mask
            neg_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device):
                    neg_targets = torch.zeros_like(
                        neg_acts[layer][..., 0], device=device
                    )
                    neg_layer_loss = probe.compute_loss(
                        neg_acts[layer],
                        neg_targets,
                        mask=neg_batch_probe_mask,  # Use probe mask
                    )
                    neg_loss += neg_layer_loss

            # Backward pass on negative examples
            neg_loss.backward(retain_graph=True)

            # Compute KL divergence of logits from base model logits
            with torch.no_grad():
                lora_model.disable_adapter_layers()
                base_neg_output = lora_model(
                    input_ids=neg_batch_input_ids,
                    # attention_mask=neg_batch_attention_mask,
                )
                lora_model.enable_adapter_layers()

            # Get logits only for masked positions
            base_logits = base_neg_output.logits[neg_batch_zero_mask]
            model_logits = neg_logits[neg_batch_zero_mask]

            kl_loss = F.kl_div(
                F.log_softmax(base_logits, dim=-1),
                F.softmax(model_logits, dim=-1),
                reduction="batchmean",
            )

            # Backward pass on KL divergence
            (kl_loss / (kl_loss.detach() + 1e-8) * kl_penalty).backward()

            # Accumulate losses
            accumulated_probe_loss += pos_loss.item() + neg_loss.item()
            accumulated_kl_loss += kl_loss.item()
            accumulated_toward_pgd_loss += (
                pgd_toward_loss if adversarial_training else 0
            )
            accumulated_probe_pgd_loss += pgd_probe_loss if adversarial_training else 0
            steps_since_last_log += 1

            # Perform optimization step after accumulating gradients
            if (i // batch_size + 1) % n_grad_accum == 0 or (
                i + batch_size
            ) >= n_examples:

                # Clip the gradients if specified
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        lora_model.parameters(), clip_grad_norm
                    )
                    all_probe_params = [
                        param
                        for probe in probes.values()
                        for param in probe.parameters()
                    ]
                    torch.nn.utils.clip_grad_norm_(all_probe_params, clip_grad_norm)

                # Optimize probes only when not using adversarial training
                if not freeze_probes_during_adversarial_training or not (
                    adversarial_training and current_step > start_adv_training_at_step
                ):
                    for optimizer in optimizers.values():
                        optimizer.step()
                        optimizer.zero_grad()

                # Optimize the adapter
                if not freeze_lora_during_warmup or not (
                    adversarial_training and current_step < start_adv_training_at_step
                ):
                    if adapter_optimizer is not None:
                        adapter_optimizer.step()
                        adapter_optimizer.zero_grad()

            current_step += 1

            if current_step % n_steps_per_logging == 0:
                avg_probe_loss = accumulated_probe_loss / steps_since_last_log
                avg_kl_loss = accumulated_kl_loss / steps_since_last_log
                avg_toward_pgd_loss = (
                    accumulated_toward_pgd_loss / steps_since_last_log
                    if adversarial_training
                    else 0
                )
                avg_probe_pgd_loss = (
                    accumulated_probe_pgd_loss / steps_since_last_log
                    if adversarial_training
                    else 0
                )
                avg_total_loss = avg_probe_loss + avg_kl_loss

                log_message = (
                    f"Step: {current_step}/{n_steps}, "
                    f"Time: {convert_seconds_to_time_str(time.time() - start_time)}, "
                    f"Avg Total Loss: {avg_total_loss:.4f}, "
                    f"Avg Probe Loss: {avg_probe_loss:.4f}, "
                    f"Avg KL Loss: {avg_kl_loss:.4f}"
                )

                if adversarial_training:
                    log_message += f", Avg Toward PGD Loss: {avg_toward_pgd_loss:.4f}"
                    log_message += f", Avg Probe PGD Loss: {avg_probe_pgd_loss:.4f}"

                print(log_message)

                # Reset accumulators
                accumulated_toward_pgd_loss = 0
                accumulated_probe_pgd_loss = 0
                accumulated_probe_loss = 0
                accumulated_kl_loss = 0
                steps_since_last_log = 0

            if current_step >= n_steps:
                continue_training_next_epoch = False
                break

            pbar.update(1)  # Update progress bar

    return probes, lora_model, info


def save_probes(probes, save_path):
    # Save a list of probes to a file
    torch.save(probes, save_path)


def load_probes(load_path):
    # Load a list of probes from a file
    return torch.load(load_path)
