import os
import subprocess

import numpy as np
import torch
from accelerate import find_executable_batch_size
from huggingface_hub import list_repo_files
from peft import AutoPeftModelForCausalLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

loaded_models = {}
loaded_tokenizers = {}


def remove_duplicates(tensor):
    # Create a dictionary to store one instance of each unique element
    seen_elements = {}

    # Iterate over the tensor and retain the first occurrence of each element
    result = []
    for idx, element in enumerate(tensor):
        if element.item() not in seen_elements:
            seen_elements[element.item()] = True
            result.append(element)

    # Convert the result list back to a tensor
    return torch.tensor(result)


def load_hf_model(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=None,
    requires_grad=False,
):
    # Check if model has already been loaded
    global loaded_models
    if model_name in loaded_models.keys():
        return loaded_models[model_name]

    # Choose attention implemention if not specified
    if attn_implementation is None:

        # Make sure that models that dont support FlashAttention aren't forced to use it
        if "gpt2" in model_name or "gemma" in model_name:
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"

    # Check if the model is peft, and load accordingly
    files = list_repo_files(model_name)
    has_adapter_config = any("adapter_config.json" in file for file in files)
    if has_adapter_config:
        model = (
            AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                device_map=device_map,
                trust_remote_code=True,
            )
            .merge_and_unload()
            .eval()
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

    # Disable model grad if we're not training
    if not requires_grad:
        model.requires_grad_(False)

    # Save and return the model
    loaded_models[model_name] = model
    return model


def load_hf_model_and_tokenizer(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=None,
    tokenizer_name=None,
    requires_grad=False,
):
    # Load the model
    model = load_hf_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        requires_grad=requires_grad,
    )

    # Get the tokenizer name if not specified
    if tokenizer_name is None:
        tokenizer_name = model.config._name_or_path

    # Check if tokenizer has already been loaded
    global loaded_tokenizers
    if tokenizer_name in loaded_tokenizers.keys():
        tokenizer = loaded_tokenizers[tokenizer_name]
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = (
            tokenizer.eos_token_id
        )  # Make sure the eos in the generation config is the same as the tokenizer

        # Save the tokenizer
        loaded_tokenizers[tokenizer_name] = tokenizer
    return model, tokenizer


def _hf_generate_with_batching(
    model, tokenizer, test_cases, batch_size, **generation_kwargs
):
    @find_executable_batch_size(starting_batch_size=batch_size)
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, generation_kwargs
        generations = []
        for i in tqdm(range(0, len(test_cases), batch_size)):
            batched_test_cases = test_cases[i : i + batch_size]
            inputs = batched_test_cases  # [template['prompt'].format(instruction=s) for s in batched_test_cases]
            inputs = tokenizer(
                inputs, return_tensors="pt", add_special_tokens=False, padding=True
            )
            inputs = inputs.to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                ).cpu()
            generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            batch_generations = [
                tokenizer.decode(o, skip_special_tokens=True).strip()
                for o in generated_tokens
            ]
            generations.extend(batch_generations)
        return generations

    return inner_generation_loop()


def dataset_generate_completions(
    partial_dataset, model, tokenizer, batch_size, *args, **kwargs
):
    # Force tokenizer to pad on the left
    initial_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Generate the completion column for all splits in the dataset
    def process_split(split):
        batch = {k: split[k] for k in split.keys()}
        gens = _hf_generate_with_batching(
            model=model,
            tokenizer=tokenizer,
            test_cases=batch["prompt"],
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        return {"completion": gens}

    # Process all splits in the dataset
    modified_dataset = partial_dataset.map(
        process_split,
        batched=True,
        batch_size=None,  # Process entire split at once
    )

    # Restore the original padding side
    tokenizer.padding_side = initial_padding_side
    return modified_dataset


def generate_completions(
    model, tokenizer, prompts, batch_size=8, verbose=False, **generation_kwargs
):
    original_paddding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    completions = []

    for i in tqdm(range(0, len(prompts), batch_size), disable=not verbose):
        batch_prompts = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs
            )

        # Remove prompt tokens from the output
        generated_tokens = outputs[:, input_ids.shape[1] :]
        batch_completions = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=False
        )
        completions.extend(batch_completions)

    tokenizer.padding_side = original_paddding_side

    return completions


def extract_submodule(module, target_path):
    # If target_path is empty, return the root module
    if not target_path:
        return module

    # Iterate over each subpart
    path_parts = target_path.split(".")
    current_module = module
    for part in path_parts:
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            raise AttributeError(f"Module has no attribute '{part}'")
    return current_module


def forward_pass_with_hooks(model, input_ids, hook_points, attention_mask=None):
    # Dictionary of hook -> activation
    activations = {}

    # Activating saving hook
    def create_hook(hook_name):
        def hook_fn(module, input, output):
            if (
                type(output) == tuple
            ):  # If the object is a transformer block, select the block output
                output = output[0]
            assert isinstance(
                output, torch.Tensor
            )  # Make sure we're actually getting the activations
            activations[hook_name] = output

        return hook_fn

    # Add a hook to every submodule we want to cache
    hooks = []
    for hook_point in hook_points:
        submodule = extract_submodule(model, hook_point)
        hook = submodule.register_forward_hook(create_hook(hook_point))
        hooks.append(hook)
    try:

        # Perform the forward pass
        with torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:

        # Remove the hooks
        for hook in hooks:
            hook.remove()
    return activations


def forward_pass_with_interventions(
    model, input_ids, hook_interventions, attention_mask=None
):
    # Dictionary to store the original outputs of intervened modules
    def create_intervention_hook(intervention_fn):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):

                # Apply intervention to the first element (usually hidden states)
                modified_first = intervention_fn(output[0])

                # Return a new tuple with the modified first element and the rest unchanged
                return (modified_first,) + output[1:]
            else:

                # If output is not a tuple, just modify and return it
                assert isinstance(output, torch.Tensor)
                return intervention_fn(output)

        return hook_fn

    # Add hooks for each intervention point
    hooks = []
    for hook_point, intervention_fn in hook_interventions.items():
        submodule = extract_submodule(model, hook_point)
        hook = submodule.register_forward_hook(
            create_intervention_hook(intervention_fn)
        )
        hooks.append(hook)
    try:

        # Perform the forward pass
        with torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:

        # Remove the hooks
        for hook in hooks:
            hook.remove()
    return outputs


def generate_with_interventions(
    model,
    input_ids,
    hook_interventions,
    max_new_tokens=20,
    attention_mask=None,
    **generation_kwargs,
):
    # Define a function to create intervention hooks
    def create_intervention_hook(intervention_fn):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):

                # Apply intervention to the first element (usually hidden states)
                modified_first = intervention_fn(output[0])

                # Return a new tuple with the modified first element and the rest unchanged
                return (modified_first,) + output[1:]
            else:

                # If output is not a tuple, just modify and return it
                assert isinstance(output, torch.Tensor)
                return intervention_fn(output)

        return hook_fn

    # List to keep track of all hooks for later removal
    hooks = []

    # Register hooks for each intervention point
    for hook_point, intervention_fn in hook_interventions.items():

        # Extract the submodule at the specified hook point
        submodule = extract_submodule(model, hook_point)

        # Register the intervention hook
        hook = submodule.register_forward_hook(
            create_intervention_hook(intervention_fn)
        )
        hooks.append(hook)
    try:

        # Perform the generation process with interventions in place
        with (
            torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype),
            torch.no_grad(),
        ):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,  # Allow for additional generation parameters
            )
    finally:

        # Ensure all hooks are removed, even if an exception occurs
        for hook in hooks:
            hook.remove()
    return outputs


def get_all_residual_acts(
    model, input_ids, attention_mask=None, batch_size=32, only_return_layers=None
):
    # Ensure the model is in evaluation mode
    model.eval()

    # Get the number of layers in the model
    num_layers = model.config.num_hidden_layers

    # Determine which layers to return
    layers_to_return = (
        set(range(num_layers))
        if only_return_layers is None
        else set(only_return_layers)
    )
    layers_to_return = {layer for layer in layers_to_return if 0 <= layer < num_layers}

    # Initialize the accumulator for hidden states
    accumulated_hidden_states = {}

    # Process input in batches
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_attention_mask = (
            attention_mask[i : i + batch_size] if attention_mask is not None else None
        )

        # Forward pass with output_hidden_states=True to get all hidden states
        with torch.no_grad():  # Disable gradient computation
            outputs = model(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )

        # The hidden states are typically returned as a tuple, with one tensor per layer
        # We want to exclude the embedding layer (first element) and get only the residual activations
        batch_hidden_states = outputs.hidden_states[1:]  # Exclude embedding layer

        # Accumulate the required layers
        for layer in layers_to_return:
            if layer not in accumulated_hidden_states:
                accumulated_hidden_states[layer] = batch_hidden_states[layer]
            else:
                accumulated_hidden_states[layer] = torch.cat(
                    [accumulated_hidden_states[layer], batch_hidden_states[layer]],
                    dim=0,
                )

    return accumulated_hidden_states


def get_all_residual_acts_unbatched(
    model, input_ids, attention_mask=None, only_return_layers=None
):
    # Get the number of layers in the model
    num_layers = model.config.num_hidden_layers

    # Determine which layers to return
    layers_to_return = (
        set(range(num_layers))
        if only_return_layers is None
        else set(only_return_layers)
    )
    layers_to_return = {layer for layer in layers_to_return if 0 <= layer < num_layers}

    # Forward pass with output_hidden_states=True to get all hidden states
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # The hidden states are typically returned as a tuple, with one tensor per layer
    # We want to exclude the embedding layer (first element) and get only the residual activations
    hidden_states = outputs.hidden_states[1:]  # Exclude embedding layer

    # Extract and return only the required layers
    return {layer: hidden_states[layer] for layer in layers_to_return}


def get_bucket_folder(bucket_path, local_path):
    # First check if the folder in GCloud has already been download.
    # If it has, just return that path.  If not, download it and then return the path
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Move up two directories
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    download_path = os.path.join(parent_dir, local_path)

    # If the path doesn't exist, download it
    if not os.path.exists(download_path):
        print("Downloading SAE weights...")
        download_from_bucket(bucket_path, download_path)
    return download_path


def download_from_bucket(bucket_path, local_path):
    # Ensure the local path exists
    os.makedirs(local_path, exist_ok=True)

    # Construct the gsutil command
    gsutil_path = os.environ.get("GSUTIL_PATH")
    if not gsutil_path:
        print("$GSUTIL_PATH is not specified, using default...")
        gsutil_path = "gsutil"
    command = [gsutil_path, "-m", "cp", "-r", bucket_path, local_path]
    try:

        # Run the gsutil command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded files from {bucket_path} to {local_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading: {e}")
        print(e.stderr)


def expand_latents(latent_indices, latent_acts, n_features):
    """
    Convert N_ctx x K indices in (0, N_features) and N_ctx x K acts into N x N_features sparse tensor
    """
    n_batch, n_pos, _ = latent_indices.shape
    expanded = torch.zeros((n_batch, n_pos, n_features), dtype=latent_acts.dtype)
    expanded.scatter_(-1, latent_indices, latent_acts)
    return expanded


def squeeze_positions(activations, aggregation="flatten", tokens=None):
    # Validate the aggregation method
    if aggregation == "max":
        # Take the maximum value across the position dimension (dim=1)
        return activations.amax(dim=1)

    elif aggregation == "mean":
        # Take the mean value across the position dimension (dim=1)
        return activations.mean(dim=1)

    elif aggregation == "flatten":
        # Merge the batch and position dimensions
        # This increases the first dimension size by a factor of sequence_length
        return activations.flatten(0, 1)

    elif aggregation == "last":
        # Select the last token's activation for each sequence in the batch
        return activations[:, -1, :]

    elif aggregation.startswith("index"):
        # index_k returns the kth token in each example. Last is index_-1
        # index_i_j returns index_i, ..., index_j concatenated
        parts = aggregation.split("_")[1:]
        try:
            if len(parts) == 1:
                idx = int(parts[0])
                return activations[:, idx, :]
            elif len(parts) == 2:
                start_idx, end_idx = parts
                if start_idx == ":":
                    start_idx = None
                else:
                    start_idx = int(start_idx)
                if end_idx == ":":
                    end_idx = None
                else:
                    end_idx = int(end_idx)

                return activations[:, start_idx:end_idx, :].reshape(
                    -1, activations.shape[-1]
                )
            else:
                raise ValueError("Invalid index format")
        except ValueError:
            raise ValueError(f"Invalid index in aggregation: {aggregation}")

    elif aggregation.startswith("aftertoken"):
        # Returns all the positions after the final occurence of the specified token
        assert tokens is not None, "Tokens must be provided for aftertoken aggregation"
        token_id = int(aggregation.split("_")[1])
        mask = (tokens == token_id).flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) == 0
        return activations[mask]

    elif aggregation.startswith("beforetoken"):
        # Returns all the positions before the final occurence of the specified token
        assert tokens is not None, "Tokens must be provided for beforetoken aggregation"
        token_id = int(aggregation.split("_")[1])
        mask = (tokens == token_id).flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) == 0
        mask = mask.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) > 0
        return activations[mask]

    elif aggregation.startswith("ontoken"):
        # Returns all the positions of the specified token
        assert tokens is not None, "Tokens must be provided for ontoken aggregation"
        token_id = int(aggregation.split("_")[1])
        mask = tokens == token_id
        return activations[mask]

    else:
        raise NotImplementedError("Invalid method")


def get_labeled(acts1, acts2, aggregation="max", acts1_tokens=None, acts2_tokens=None):
    # Use squeeze_positions to aggregate across the position dimension
    acts1 = squeeze_positions(acts1, aggregation, tokens=acts1_tokens)
    acts2 = squeeze_positions(acts2, aggregation, tokens=acts2_tokens)

    # Combine the features from both splits
    input_acts = torch.cat([acts1, acts2], dim=0)

    # Create labels: 0 for split1, 1 for split2
    labels = torch.cat(
        [
            torch.zeros(acts1.shape[0], dtype=torch.long),
            torch.ones(acts2.shape[0], dtype=torch.long),
        ]
    )
    return input_acts, labels


def normalize_last_dim(tensor):
    # Compute the L2 norm along the last dimension
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)

    # Divide the tensor by the norm
    # We add a small epsilon to avoid division by zero
    normalized_tensor = tensor / (norm + 1e-8)
    return normalized_tensor


def get_last_true_indices(mask):
    # Multiply positions by mask and take max for each row
    positions = torch.arange(mask.size(1), device=mask.device)
    return (positions * mask).argmax(dim=1) + 1


def get_valid_indices(tokens, only_return_on_tokens_between):
    # Get indices of tokens between start and end tokens or predicates

    valid_indices = []
    start_token, end_token = only_return_on_tokens_between
    include_indices = False

    def match(index, token, tokens, matcher):
        if callable(matcher):
            return matcher(index, token, tokens)
        else:
            return token == matcher

    for i, token in enumerate(tokens):
        if match(i, token, tokens, start_token):
            include_indices = True
        elif match(i, token, tokens, end_token):
            include_indices = False
        elif include_indices:
            valid_indices.append(i)

    return valid_indices


def get_valid_token_mask(tokens, only_return_on_tokens_between):
    # Get a mask of tokens between start and end tokens or predicates
    if tokens.dim() not in (1, 2):
        raise ValueError("Input tensor must be 1D or 2D")

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    batch_size, seq_length = tokens.shape
    start_token, end_token = only_return_on_tokens_between

    def match(seq_idx, token, tokens, matcher):
        if callable(matcher):
            return matcher(seq_idx, token.item(), tokens)
        else:
            return token.item() == matcher

    # Initialize the mask with zeros
    mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=tokens.device)

    for i in range(batch_size):
        include_indices = False
        for j in range(seq_length):
            token = tokens[i, j]
            if match(j, token, tokens[i], start_token):
                include_indices = True
            elif match(j, token, tokens[i], end_token):
                include_indices = False
            elif include_indices:
                mask[i, j] = True

    return mask.squeeze(0) if tokens.dim() == 1 else mask


def convert_float16(obj):
    if isinstance(obj, dict):
        return {k: convert_float16(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_float16(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_float16(v) for v in obj)
    elif isinstance(obj, np.float16):
        return float(obj)
    elif isinstance(obj, np.ndarray) and obj.dtype == np.float16:
        return obj.astype(float)
    elif isinstance(obj, torch.Tensor) and obj.dtype == torch.float16:
        return obj.float()
    else:
        return obj


def convert_seconds_to_time_str(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = seconds // 3600
        seconds = seconds % 3600
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds}s"


def process_data(prompts, targets, tokenizer, batch_size=None):
    tokenized_prompts = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    tokenized_targets = [
        tokenizer.encode(target, add_special_tokens=False) for target in targets
    ]

    if batch_size is None:
        batch_size = len(prompts)

    max_length = max(
        len(tokenized_prompts[i] + tokenized_targets[i]) for i in range(batch_size)
    )

    adv_tokens = torch.zeros((batch_size, max_length), dtype=torch.long)
    adv_tokens.fill_(tokenizer.pad_token_id)
    prompt_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    target_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    for i in tqdm(range(batch_size), desc="Tokenizing"):
        combined = tokenized_prompts[i] + tokenized_targets[i]
        adv_tokens[i, : len(combined)] = torch.tensor(combined)
        prompt_mask[i, : len(tokenized_prompts[i])] = True
        target_mask[i, len(tokenized_prompts[i]) : len(combined)] = True

    return adv_tokens, prompt_mask, target_mask


def process_dataset_rows(rows, tokenizer=None, batch_size=None):
    # Process a single row of the dataset
    prompts = rows["prompt"]
    targets = rows["completion"]
    adv_tokens = []
    prompt_mask = []
    target_mask = []
    if batch_size is None:
        batch_size = len(prompts)
    for i in range(len(prompts) // batch_size):
        _adv_tokens, _prompt_mask, _target_mask = process_data(
            prompts, targets, tokenizer, batch_size=batch_size
        )
        adv_tokens.append(_adv_tokens)
        prompt_mask.append(_prompt_mask)
        target_mask.append(_target_mask)
    rows["adv_tokens"] = adv_tokens
    rows["prompt_mask"] = prompt_mask
    rows["target_mask"] = target_mask
    return {
        "tokens": adv_tokens,
        "prompt_mask": prompt_mask,
        "target_mask": target_mask,
    }


def remove_ood_activations_np(X, y=None, threshold_multiplier=3.5, verbose=True):
    # Compute norms of activations
    norms = np.linalg.norm(X, axis=1)

    # Compute median and median absolute deviation (MAD)
    median_norm = np.median(norms)
    mad = np.median(np.abs(norms - median_norm))

    # Define a threshold for filtering using modified Z-score
    # A modified Z-score of 3.5 or higher is often considered an extreme outlier
    threshold = median_norm + threshold_multiplier * 1.4826 * mad

    # Compute the mask for filtering
    mask = norms < threshold

    if verbose:
        # Calculate diagnostics
        removed_norms = norms[~mask]
        kept_norms = norms[mask]

        if len(removed_norms) > 0:
            avg_removed_norm = np.mean(removed_norms)
        else:
            avg_removed_norm = 0

        max_kept_norm = np.max(kept_norms)

        print(f"Average norm of removed activations: {avg_removed_norm:.4f}")
        print(f"Maximum norm of non-removed activations: {max_kept_norm:.4f}")
        print(f"Number of activations removed: {np.sum(~mask)}")
        print(f"Number of activations kept: {np.sum(mask)}")

    # Apply the filter
    if y is not None:
        return X[mask], y[mask]
    else:
        return X[mask]


def remove_ood_activations(X, y=None, threshold_multiplier=3.5, verbose=True):
    # Compute norms of activations
    norms = torch.norm(X, dim=1)

    # Compute median and median absolute deviation (MAD)
    median_norm = torch.median(norms)
    mad = torch.median(torch.abs(norms - median_norm))

    # Define a threshold for filtering using modified Z-score
    # A modified Z-score of 3.5 or higher is often considered an extreme outlier
    threshold = median_norm + threshold_multiplier * 1.4826 * mad

    # Compute the mask for filtering
    mask = norms < threshold

    if verbose:
        # Calculate diagnostics
        removed_norms = norms[~mask]
        kept_norms = norms[mask]

        if len(removed_norms) > 0:
            avg_removed_norm = torch.mean(removed_norms).item()
        else:
            avg_removed_norm = 0

        max_kept_norm = torch.max(kept_norms).item()

        print(f"Average norm of removed activations: {avg_removed_norm:.4f}")
        print(f"Maximum norm of non-removed activations: {max_kept_norm:.4f}")
        print(f"Number of activations removed: {(~mask).sum().item()}")
        print(f"Number of activations kept: {mask.sum().item()}")

    # Apply the filter while preserving gradients
    if y is not None:
        return X[mask], y[mask]
    else:
        return X[mask]


def zero_nan_grads(model):
    flag = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                flag = True
                p.grad[torch.isnan(p.grad)] = 0.0
    return flag


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif hasattr(obj, "__dict__"):
        return convert_to_serializable(obj.__dict__)
    return obj
