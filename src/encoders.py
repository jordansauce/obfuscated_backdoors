import json
import os
import warnings

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from tqdm.auto import tqdm
from transformer_lens import utils as tl_utils

from .helper_classes import *
from .sae import Sae
from .utils import *


class SparseAutoencoder:
    """
    SAE Interface to abstract away all the details
    """

    def __init__(self, model, tokenizer, hook_name, n_features, max_k):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_name = hook_name
        self.n_features = n_features
        self.max_k = max_k
        self.hook_names = [
            hook_name,
        ]

    def reconstruct(self, acts):
        raise NotImplementedError()

    def encode(self, acts):
        raise NotImplementedError()

    def get_codebook(self, hook_name):
        raise NotImplementedError()

    def _zero_bos_acts(self, input_ids, acts):
        # Zero out the BOS token activations
        bos_mask = input_ids == self.tokenizer.bos_token_id
        acts[bos_mask] = 0
        return acts

    @torch.inference_mode()
    def featurize(self, tokens, masks=None):
        """
        Returns a dictionary with hook_name as key and a tuple of two B x P x k tensors of feature activations as value
        Nonactivating features will be a zero
        """

        # Initialize output tensor
        n_batch, n_pos = tokens.shape

        # Calculate SAE features
        with torch.no_grad():

            # Do a forward pass to get model acts
            model_acts_layer = forward_pass_with_hooks(
                model=self.model,
                input_ids=tokens.to(self.model.device),
                attention_mask=(
                    masks.to(self.model.device) if masks is not None else None
                ),
                hook_points=[self.hook_name],
            )[self.hook_name]

            # Calculate the SAE features
            top_indices, top_acts = self.encode(model_acts_layer.flatten(0, 1))
            latent_indices = top_indices.reshape(n_batch, n_pos, -1).cpu()
            latent_acts = top_acts.reshape(n_batch, n_pos, -1).cpu()
            latent_acts = self._zero_bos_acts(tokens, latent_acts)
        return {self.hook_name: (latent_indices, latent_acts)}

    @torch.inference_mode()
    def batched_featurize(self, tokens, masks=None, batch_size=None):
        """
        Batched version of featurize
        """
        if batch_size is None:
            batch_size = tokens.shape[0]
        minibatches_tokens = tokens.split(batch_size)
        minibatches_masks = (
            masks.split(batch_size)
            if masks is not None
            else [None] * len(minibatches_tokens)
        )
        all_latent_indices = []
        all_latent_acts = []

        # Featurize every single minibatch
        for minibatch_tokens, minibatch_masks in tqdm(
            zip(minibatches_tokens, minibatches_masks)
        ):
            result = self.featurize(minibatch_tokens, minibatch_masks)
            latent_indices, latent_acts = result[self.hook_name]
            all_latent_indices.append(latent_indices)
            all_latent_acts.append(latent_acts)

        # Concatenate all results
        latent_indices = torch.cat(all_latent_indices, dim=0)
        latent_acts = torch.cat(all_latent_acts, dim=0)
        return {self.hook_name: (latent_indices, latent_acts)}

    def featurize_text(self, text, batch_size=None, max_length=512):
        """
        Tokenize and featurize the text
        """
        max_length = min(self.tokenizer.model_max_length, max_length)
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",  # Pad to the longest sequence in the batch
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        if batch_size is None:
            return self.featurize(input_ids, attention_mask)
        else:
            return self.batched_featurize(input_ids, attention_mask, batch_size)

    @torch.inference_mode()
    def get_model_residual_acts(
        self,
        text,
        batch_size=None,
        max_length=1024,
        return_tokens=False,
        use_memmap=None,
        only_return_layers=None,
        only_return_on_tokens_between=None,
        verbose=True,
    ):
        # Ensure max_length doesn't exceed the model's maximum
        max_length = min(self.tokenizer.model_max_length, max_length)

        # Tokenize the input text
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",  # Pad to the longest sequence in the batch
            max_length=max_length,
            truncation=True,  # Truncate sequences longer than max_length
            return_attention_mask=True,  # Get attention mask to handle padding
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Apply the only_return_on_tokens_between mask if provided
        if only_return_on_tokens_between is not None:
            only_return_mask = get_valid_token_mask(
                input_ids, only_return_on_tokens_between
            )
            zero_positions_mask = attention_mask.clone()
            zero_positions_mask[~only_return_mask] = 0
        else:
            zero_positions_mask = attention_mask

        # If batch_size is not specified, process all input at once
        if batch_size is None:
            batch_size = input_ids.size(0)

        # Get the number of layers and hidden dimension
        num_layers = self.model.config.num_hidden_layers
        hidden_dim = self.model.config.hidden_size

        # Calculate the full shape of activations
        full_shape = (input_ids.size(0), input_ids.size(1), hidden_dim)

        # Initialize memmaps if a file path is provided
        if use_memmap:
            memmap_dir = os.path.dirname(use_memmap)
            os.makedirs(memmap_dir, exist_ok=True)

            # Initialize metadata dictionary
            metadata = {
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "shape": full_shape,
                "dtype": "float16",
                "files": {},
            }

            # Create memmap for tokens if return_tokens is True
            if return_tokens:
                tokens_memmap_file = f"{use_memmap}_tokens.dat"
                tokens_memmap = np.memmap(
                    tokens_memmap_file, dtype="int32", mode="w+", shape=input_ids.shape
                )
                tokens_memmap[:] = input_ids.numpy()
                metadata["tokens_file"] = os.path.basename(tokens_memmap_file)
                metadata["tokens_shape"] = input_ids.shape
                metadata["tokens_dtype"] = "int32"

            # Create memmaps for each layer with the correct shape
            layer_memmaps = {}
            layers_to_return = (
                range(num_layers) if only_return_layers is None else only_return_layers
            )
            for layer in layers_to_return:
                memmap_file = f"{use_memmap}_residual_act_layer_{layer}.dat"
                memmap = np.memmap(
                    memmap_file, dtype="float16", mode="w+", shape=full_shape
                )
                layer_memmaps[layer] = memmap
                metadata["files"][f"layer_{layer}"] = os.path.basename(memmap_file)

            # Save metadata
            metadata_file = f"{use_memmap}_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        else:
            # Pre-allocate the full tensor for required layers
            layers_to_return = (
                range(num_layers) if only_return_layers is None else only_return_layers
            )
            all_residual_acts = {
                layer: torch.empty(
                    (input_ids.size(0), input_ids.size(1), hidden_dim),
                    dtype=torch.float16,
                    device="cpu",
                )
                for layer in layers_to_return
            }

        # Process input in batches
        for i in tqdm(range(0, input_ids.size(0), batch_size), disable=not verbose):
            # Extract batch and move to device (GPU/CPU)
            batch_input_ids = input_ids[i : i + batch_size].to(self.model.device)
            batch_attention_mask = attention_mask[i : i + batch_size].to(
                self.model.device
            )
            batch_zero_positions_mask = zero_positions_mask[i : i + batch_size].to(
                self.model.device
            )  # Masked attention mask for only_return_on_tokens_between

            # Get residual activations for the batch
            batch_residual_acts = get_all_residual_acts_unbatched(
                self.model, batch_input_ids, batch_attention_mask, only_return_layers
            )

            # Apply attention mask to the activations and move to CPU
            masked_batch_residual_acts = {
                layer: (act * batch_zero_positions_mask.unsqueeze(-1).to(act.dtype))
                .cpu()
                .to(torch.float16)
                for layer, act in batch_residual_acts.items()
            }

            if use_memmap:
                # Write the batch data to the memmaps
                for layer, act in masked_batch_residual_acts.items():
                    layer_memmaps[layer][i : i + batch_size] = act.numpy()
            else:
                # Directly assign the batch results to the pre-allocated tensors
                for layer, act in masked_batch_residual_acts.items():
                    all_residual_acts[layer][i : i + batch_size] = act

        # Return the residual activations
        if return_tokens:
            if use_memmap:
                return layer_memmaps, tokens
            else:
                return all_residual_acts, tokens
        else:
            if use_memmap:
                return layer_memmaps
            else:
                return all_residual_acts

    @torch.inference_mode()
    def get_examples_from_generations(
        self, generations, max_length=1024, *args, **kwargs
    ):
        # Tokenize the generations
        tokenizer_output = self.tokenizer(
            generations,
            return_tensors="pt",
            padding="longest",  # Pad to the longest sequence in the batch
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        tokens = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        # Get the SAE activations for the generations
        result = self.featurize_text(
            generations, max_length=max_length, *args, **kwargs
        )

        # Create example classes
        examples = []
        for i in range(len(tokens)):

            # Get the non-padding token positions
            non_padding_positions = attention_mask[i].bool()

            # Filter out padding tokens
            non_padded_tokens = tokens[i][non_padding_positions]

            # Filter out padding activations for each hook
            non_padded_latent_data = {
                hook: (
                    result[hook][0][i][non_padding_positions],
                    result[hook][1][i][non_padding_positions],
                )
                for hook in result
            }
            examples.append(
                Example(
                    tokens=non_padded_tokens,
                    tokenizer=self.tokenizer,
                    latent_data=non_padded_latent_data,
                )
            )
        return examples

    def format_inputs(self, prompt, system_prompt=None):
        # Format input for language models using the tokenizer's chat template.
        def format_single_input(single_prompt):
            # Format a single prompt with optional system message using the tokenizer's chat template.
            # Prepare messages in the format expected by chat models
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add user prompt
            messages.append({"role": "user", "content": single_prompt})
            try:

                # Use the tokenizer's chat template
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except AttributeError:
                warnings.warn(
                    "The provided tokenizer does not have the 'apply_chat_template' method. "
                    "Falling back to a simple format. This may not be optimal for all models.",
                    UserWarning,
                )

                # Simple fallback format
                formatted = ""
                if system_prompt:
                    formatted += f"{system_prompt}\n"
                formatted += f"{single_prompt}"

                # Add BOS token if available
                bos_token = getattr(self.tokenizer, "bos_token", "")
                return f"{bos_token}{formatted}"

        # Handle input based on whether it's a single string or a list of strings
        if isinstance(prompt, str):
            return format_single_input(prompt)
        elif isinstance(prompt, list):
            return [format_single_input(p) for p in prompt]
        else:
            raise ValueError("prompt must be either a string or a list of strings")

    def sample_generations(
        self,
        prompts,
        format_inputs=True,
        batch_size=4,
        system_prompt=None,
        return_examples=True,
        **generation_kwargs,
    ):
        # Make sure prompts is a list
        if not isinstance(prompts, list):
            print(f"Prompts must be a list but is {type(prompts)}, wrapping in a list")
            prompts = [prompts]

        # Format inputs if requested
        if format_inputs:
            prompts = self.format_inputs(prompts, system_prompt)
        all_generations = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize inputs
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_return_sequences=1,
                    **generation_kwargs,
                )

            # Decode generated text
            batch_generations = self.tokenizer.batch_decode(outputs)
            batch_generations = [
                gen.replace(self.tokenizer.pad_token, "") for gen in batch_generations
            ]
            all_generations.extend(batch_generations)

        if return_examples:
            return self.get_examples_from_generations(all_generations)
        return all_generations

    def _fix_input_shape(self, acts):
        if len(acts.shape) == 0:
            raise Exception("0-dimensional input")
        elif len(acts.shape) == 1:
            acts = acts[None, :]
        return acts

    def __repr__(self):
        return f"{self.__class__.__name__}(hook_name={self.hook_name}, n_features={self.n_features}, max_k={self.max_k})"


class LanguageModelWrapper(SparseAutoencoder):
    # Wrapper class for language models

    def __init__(self, model, tokenizer):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            hook_name="",
            n_features=None,
            max_k=None,
        )

    def featurize(self, tokens, masks=None):
        raise NotImplementedError()

    def sample_generations(
        self,
        prompts,
        format_inputs=True,
        batch_size=4,
        system_prompt=None,
        **generation_kwargs,
    ):
        # Never convert the generations to SAE Examples
        return super().sample_generations(
            prompts,
            format_inputs,
            batch_size,
            system_prompt,
            False,
            **generation_kwargs,
        )


class SparseAutoencoderCollection(SparseAutoencoder):
    """
    Takes in a list of SparseAutoencoders and treats them as a group.
    Allows for aggregating common attributes across the collection.
    """

    def __init__(self, encoder_list):
        if not encoder_list:
            raise ValueError("encoder_list must not be empty")
        self.encoders = encoder_list
        self.encoder_dict = {encoder.hook_name: encoder for encoder in encoder_list}

        # Set the attributes that are supposed to be common in all of the encoders
        common_attributes = ["model", "tokenizer", "n_features", "max_k"]
        self._set_common_attributes(common_attributes)

        # Create encoder dictionary that's hook -> encoder
        self.encoder = {encoder.hook_name: encoder for encoder in encoder_list}

        # Collect all hook names
        self.hook_names = list(self.encoder.keys())

    def _set_common_attributes(self, common_attributes):
        # Sets the common attributes
        for attr in common_attributes:
            values = [
                getattr(encoder, attr)
                for encoder in self.encoders
                if hasattr(encoder, attr)
            ]
            if not values:
                continue  # Skip if no encoder has this attribute
            if len(set(values)) > 1:
                raise ValueError(
                    f"Inconsistent values for attribute '{attr}' across encoders: {values}"
                )
            setattr(self, attr, values[0])

    def get_codebook(self, hook_name):
        return self.encoder[hook_name].get_codebook(hook_name)

    @torch.inference_mode()
    def featurize(self, tokens, masks=None):
        """
        Returns a dictionary of hook_name -> (latent_indices, latent_acts) for each encoder
        """
        n_batch, n_pos = tokens.shape
        results = {}
        with torch.no_grad():

            # Do a forward pass to get model acts for all hooks
            model_acts = forward_pass_with_hooks(
                model=self.model,
                input_ids=tokens.to(self.model.device),
                attention_mask=(
                    masks.to(self.model.device) if masks is not None else None
                ),
                hook_points=self.hook_names,
            )

            # Calculate the SAE features for each encoder
            for hook_name, encoder in self.encoder.items():
                model_acts_layer = model_acts[hook_name]
                top_indices, top_acts = encoder.encode(model_acts_layer.flatten(0, 1))
                latent_indices = top_indices.reshape(n_batch, n_pos, -1).cpu()
                latent_acts = top_acts.reshape(n_batch, n_pos, -1).cpu()
                latent_acts = self._zero_bos_acts(tokens, latent_acts)
                results[hook_name] = (latent_indices, latent_acts)
        return results

    @torch.inference_mode()
    def batched_featurize(self, tokens, masks=None, batch_size=None):
        """
        Batched version of featurize
        """
        if batch_size is None:
            batch_size = tokens.shape[0]
        minibatches_tokens = tokens.split(batch_size)
        minibatches_masks = (
            masks.split(batch_size)
            if masks is not None
            else [None] * len(minibatches_tokens)
        )
        all_results = {hook: ([], []) for hook in self.hook_names}

        # Featurize every single minibatch
        for minibatch_tokens, minibatch_masks in tqdm(
            zip(minibatches_tokens, minibatches_masks)
        ):
            batch_results = self.featurize(minibatch_tokens, minibatch_masks)
            for hook, (indices, acts) in batch_results.items():
                all_results[hook][0].append(indices)
                all_results[hook][1].append(acts)

        # Concatenate all results
        for hook in self.hook_names:
            all_results[hook] = (
                torch.cat(all_results[hook][0], dim=0),
                torch.cat(all_results[hook][1], dim=0),
            )
        return all_results

    def reconstruct(self, acts):
        raise Exception(
            "Collection does not support reconstruct, use featurize instead"
        )

    def encode(self, acts):
        raise Exception("Collection does not support encode, use featurize instead")

    def __repr__(self):
        return f"{self.__class__.__name__}(encoders={self.encoders})"


class GenericSaeModule(nn.Module):
    """
    Module to hold SAE parameters
    """

    def __init__(self, d_model, d_sae):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_model))
        )
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_model, d_sae))
        )
        self.b_dec = nn.Parameter(torch.zeros(d_model))


class EleutherSparseAutoencoder(SparseAutoencoder):
    """
    Wrapper class for EleutherAI Top-K SAEs
    """

    def __init__(self, model, tokenizer, encoder, hook_name):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_name,
            n_features=encoder.encoder.out_features if encoder is not None else None,
            max_k=encoder.cfg.k if encoder is not None else None,
        )
        self.encoder = encoder

    def reconstruct(self, acts):
        if self.encoder is None:
            raise ValueError("This model does not have an SAE")
        # Encode the acts in SAE space and decode back
        acts = self._fix_input_shape(acts)
        return self.encoder(acts).sae_out

    def encode(self, acts):
        if self.encoder is None:
            raise ValueError("This model does not have an SAE")
        # Encode the acts in SAE space
        acts = self._fix_input_shape(acts)
        out = self.encoder.encode(acts)
        return out.top_indices, out.top_acts

    def get_codebook(self, hook_name):
        if self.encoder is None:
            raise ValueError("This model does not have an SAE")
        return self.encoder.W_dec

    @staticmethod
    def load_llama3_sae(
        layer,
        instruct=True,
        v2=False,
        other_model_tokenizer=(None, None),
        device="cuda",
        *args,
        **kwargs,
    ):
        # Loading LLaMa3 SAEs trained by Nora Belrose
        # Load the model from huggingface
        if other_model_tokenizer[0] is not None:
            print("A model and tokenizer were provided, using those instead")
            model, tokenizer = other_model_tokenizer
        else:
            model_name = (
                "meta-llama/Meta-Llama-3-8B-Instruct"
                if instruct
                else "meta-llama/Meta-Llama-3-8B"
            )
            model, tokenizer = load_hf_model_and_tokenizer(
                model_name, device_map=device
            )

        # Load SAE using Eleuther library
        if layer is None:
            sae = None
        else:
            sae_name = (
                "EleutherAI/sae-llama-3-8b-32x-v2"
                if v2
                else "EleutherAI/sae-llama-3-8b-32x"
            )
            sae = Sae.load_from_hub(
                sae_name, hookpoint=f"layers.{layer}", device=device
            )
        return EleutherSparseAutoencoder(
            model=model,
            tokenizer=tokenizer,
            encoder=sae,
            hook_name=f"model.layers.{layer}",  # The SAE reads in the output of this block
            *args,
            **kwargs,
        )

    @staticmethod
    def load_pythia_sae(
        layer,
        model_size="160m",
        deduped=True,
        other_model_tokenizer=(None, None),
        device="cuda",
        *args,
        **kwargs,
    ):
        # Loading Pythia SAEs trained by EleutherAI
        assert model_size in [
            "70m",
            "160m",
        ], "Residual stream SAEs are only available for 70m and 160m models"

        # Load the model from huggingface
        if other_model_tokenizer[0] is not None:
            print("A model and tokenizer were provided, using those instead")
            model, tokenizer = other_model_tokenizer
        else:
            model_name = (
                f"EleutherAI/pythia-{model_size}-deduped"
                if deduped
                else f"EleutherAI/pythia-{model_size}"
            )
            model, tokenizer = load_hf_model_and_tokenizer(model_name)

        # Load SAE using Eleuther library
        if layer is None:
            sae = None
        else:
            sae_name = (
                f"EleutherAI/sae-pythia-{model_size}-deduped-32k"
                if deduped
                else f"EleutherAI/sae-pythia-{model_size}-32k"
            )
            sae = Sae.load_from_hub(
                sae_name, hookpoint=f"layers.{layer}", device=device
            )
        return EleutherSparseAutoencoder(
            model=model,
            tokenizer=tokenizer,
            encoder=sae,
            hook_name=f"model.layers.{layer}",  # The SAE reads in the output of this block
            *args,
            **kwargs,
        )


class DeepmindSparseAutoencoder(SparseAutoencoder):
    """
    Wrapper class for DeepMind JumpRELU SAE
    """

    def __init__(self, model, tokenizer, encoder, hook_name, max_k_features=192):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_name,
            n_features=encoder.W_enc.shape[1] if encoder is not None else None,
            max_k=max_k_features,
        )
        self.encoder = encoder

    def reconstruct(self, acts):
        if self.encoder is None:
            raise ValueError("This model does not have an SAE")
        # Encode the acts in SAE space
        acts = self._fix_input_shape(acts)
        sae_latents = acts @ self.encoder.W_enc + self.encoder.b_enc
        sae_latents = torch.relu(sae_latents) * (sae_latents > self.encoder.threshold)
        return sae_latents @ self.encoder.W_dec + self.encoder.b_dec

    def encode(self, acts):
        if self.encoder is None:
            raise ValueError("This model does not have an SAE")
        # Encode the acts in SAE space
        acts = self._fix_input_shape(acts)
        sae_latents = acts @ self.encoder.W_enc + self.encoder.b_enc
        sae_latents = torch.relu(sae_latents) * (sae_latents > self.encoder.threshold)
        top_sae_latents = sae_latents.topk(self.max_k, dim=-1, sorted=False)
        return top_sae_latents.indices, top_sae_latents.values

    def get_codebook(self, hook_name):
        return self.encoder.W_dec

    @staticmethod
    def load_npz_weights(weight_path, dtype, device):
        state_dict = {}
        with np.load(weight_path) as data:
            for key in data.keys():
                state_dict_key = key
                if state_dict_key.startswith("w_"):

                    # I think that communication between Joseph and us
                    # caused a misconception here, so I'll correct it.
                    state_dict_key = "W_" + state_dict_key[2:]
                if dtype is not None:
                    state_dict[key] = torch.tensor(data[key]).to(dtype=dtype).to(device)
                else:
                    state_dict[key] = torch.tensor(data[key]).to(dtype=dtype).to(device)
        return state_dict

    @staticmethod
    def load_gemma2_sae(
        layer,
        l0,
        width=131072,
        instruct=True,
        other_model_tokenizer=(None, None),
        *args,
        **kwargs,
    ):
        # Loading Gemma 2 9b SAEs by Google DeepMind
        # Load the model from huggingface
        if other_model_tokenizer[0] is not None:
            print("A model and tokenizer were provided, using those instead")
            model, tokenizer = other_model_tokenizer
        else:
            model_name = "google/gemma-2-9b-it" if instruct else "google/gemma-2-9b"
            model, tokenizer = load_hf_model_and_tokenizer(model_name)

        # Load sae weights into module
        if layer is None:
            sae = None
        else:
            # Download/Load the sae
            repo_id = "google/gemma-scope-9b-pt-res"
            filename = f"layer_{layer}/width_{width//10**3}k/average_l0_{l0}/params.npz"
            sae_path = hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type="model"
            )

            # Load the weights
            sae = (
                GenericSaeModule(d_model=model.config.hidden_size, d_sae=width)
                .cuda()
                .to(torch.bfloat16)
            )
            sae.load_state_dict(
                DeepmindSparseAutoencoder.load_npz_weights(
                    sae_path, torch.bfloat16, "cuda"
                )
            )

        return DeepmindSparseAutoencoder(
            model=model,
            tokenizer=tokenizer,
            encoder=sae,
            hook_name=f"model.layers.{layer}",  # The SAE reads in the output of this block
            *args,
            **kwargs,
        )
