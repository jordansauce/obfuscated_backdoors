import json
import os

import numpy as np
import torch
import transformer_lens.utils as tl_utils
from tqdm.auto import tqdm
from transformer_lens import utils as tl_utils

from .helper_classes import *
from .utils import *


class FeatureDatabase:
    """
    Get SAE Feature activations over whole dataset and save to disk
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.feature_data = None
        self.hook_names = encoder.hook_names
        self.features = {
            hook_name: [
                Feature(feature_id, hook_name, self)
                for feature_id in range(encoder.n_features)
            ]
            for hook_name in self.hook_names
        }
        self.k = encoder.max_k
        self.n = encoder.n_features

    def process_dataset(
        self,
        dataset,
        n_examples,
        folder_name,
        seed=42,
        example_seq_len=128,
        batch_size=128,
        text_column="text",
    ):
        # Save dataset config to folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        config_file_name = os.path.join(folder_name, "config.json")
        with open(config_file_name, "w") as f:
            json.dump(
                {
                    "n_examples": n_examples,
                    "example_seq_len": example_seq_len,
                    "hook_names": self.hook_names,
                },
                f,
            )

        # Chunk and tokenize the dataset
        dataset = dataset.shuffle(seed)
        tokenized_data = tl_utils.tokenize_and_concatenate(
            dataset.select(range(n_examples)),
            self.encoder.tokenizer,
            column_name=text_column,
            max_length=example_seq_len,
        )
        all_tokens = tokenized_data["tokens"][:n_examples]
        all_attention_masks = (all_tokens != self.encoder.tokenizer.pad_token_id).long()
        print(f"Processing {n_examples} contexts...")

        # Cache sae feature activations over tokens
        self._build_memmap(
            all_tokens,
            all_attention_masks,
            n_examples,
            folder_name,
            example_seq_len,
            batch_size,
        )

    def _build_memmap(
        self,
        all_tokens,
        all_attention_masks,
        n_examples,
        folder_name,
        example_seq_len,
        batch_size,
    ):
        # Create memmap in directory for each hook
        bytes_per_row = 8 + self.k * 6  # (4 bytes for index, 2 bytes for activation)
        self.feature_data = {}
        for hook_name in self.hook_names:
            self.feature_data[hook_name] = np.memmap(
                os.path.join(folder_name, f"sae_feature_data_{hook_name}.memmap"),
                dtype="uint8",
                mode="w+",
                shape=(n_examples, example_seq_len, bytes_per_row),
            )

        # Generate all examples and save to disk
        minibatches_tokens = all_tokens.split(batch_size)
        minibatches_masks = all_attention_masks.split(batch_size)
        ctr = 0
        for minibatch_tokens, minibatch_masks in tqdm(
            zip(minibatches_tokens, minibatches_masks), desc="Caching SAE Activations"
        ):

            # Get sae features
            new_feat_activations = self.encoder.featurize(
                minibatch_tokens, minibatch_masks
            )

            # Save the SAE feature activations for each hook
            N, P = minibatch_tokens.shape[0], example_seq_len
            tokens = minibatch_tokens.numpy().astype(np.int64)
            for hook_name, (latent_inds, latent_acts) in new_feat_activations.items():
                self.feature_data[hook_name][ctr : ctr + N, :, 0:8] = tokens.view(
                    np.uint8
                ).reshape(N, P, 8)
                self.feature_data[hook_name][ctr : ctr + N, :, 8 : 8 + 4 * self.k] = (
                    latent_inds.numpy()
                    .astype(np.int32)
                    .view(np.uint8)
                    .reshape(N, P, 4 * self.k)
                )
                self.feature_data[hook_name][ctr : ctr + N, :, 8 + 4 * self.k :] = (
                    latent_acts.numpy()
                    .astype(np.float16)
                    .view(np.uint8)
                    .reshape(N, P, 2 * self.k)
                )
                self.feature_data[hook_name].flush()
            ctr += N

    def load_from_disk(self, folder_name):
        # Load dataset config
        if not os.path.exists(folder_name):
            raise Exception("Dataset not created!!!")
        config_file_name = os.path.join(folder_name, "config.json")
        with open(config_file_name, "r") as f:
            config = json.load(f)
        n_examples = config["n_examples"]
        example_seq_len = config["example_seq_len"]
        self.hook_names = config["hook_names"]

        # Create memmap in directory for each hook
        bytes_per_row = 8 + self.k * 6  # (4 bytes for index, 2 bytes for activation)
        self.feature_data = {}
        for hook_name in self.hook_names:
            self.feature_data[hook_name] = np.memmap(
                os.path.join(folder_name, f"sae_feature_data_{hook_name}.memmap"),
                dtype="uint8",
                mode="r",
                shape=(n_examples, example_seq_len, bytes_per_row),
            )

    def assert_loaded(self):
        # Check if the memmap has been loaded yet
        if self.feature_data is None:
            raise Exception("Memmap has not been loaded yet")

    def load_example(self, example_id):
        self.assert_loaded()

        # Extract tokens, latent indices, and latent activations for each hook
        result = {}
        for hook_name, data in self.feature_data.items():
            tokens = data[example_id, :, 0:8].view(np.int64).flatten()
            latent_indices = data[example_id, :, 8 : 8 + 4 * self.k].view(np.int32)
            latent_acts = data[example_id, :, 8 + 4 * self.k :].view(np.float16)
            result[hook_name] = (
                torch.from_numpy(latent_indices),
                torch.from_numpy(latent_acts),
            )

        # Load example from feature activations
        return Example(
            tokens=torch.from_numpy(tokens),
            tokenizer=self.encoder.tokenizer,
            latent_data=result,
        )

    def _get_tiv_parts(self, hook_name, zero_bos_acts=True, load_to_memory=False):
        self.assert_loaded()

        # Separates data into tokens, indices, and values
        assert hook_name in self.hook_names
        data = self.feature_data[hook_name]
        tokens = data[:, :, 0:8].view(np.int64).squeeze(-1)
        indices = data[:, :, 8 : 8 + 4 * self.k].view(np.int32)
        values = data[:, :, 8 + 4 * self.k :].view(np.float16)

        # Load data to memory if requested
        if load_to_memory:
            tokens = np.array(tokens)
            indices = np.array(indices)
            values = np.array(values)

        # Zero out the feature activations on the bos tokens
        if zero_bos_acts:
            bos_mask = tokens == self.encoder.tokenizer.bos_token_id

            # Create a copy of values before modifying if it's not already in memory
            if not load_to_memory:
                values = np.array(values, copy=True)
            values[bos_mask] = 0  # or -1 if you prefer
        return (tokens, indices, values)

    def get_common_features(self, hook_name, k=10, chunk_size=1024):
        # Get ranked top most higly activating features in the database
        self.assert_loaded()
        tokens, indices, values = self._get_tiv_parts(hook_name, zero_bos_acts=True)
        n_batch, n_pos, n_k = indices.shape
        max_activations = np.zeros(self.n, dtype=values.dtype)
        for i in tqdm(range(0, n_batch, chunk_size)):
            chunk_indices = indices[i : i + chunk_size]
            chunk_values = values[i : i + chunk_size]

            # Flatten the chunk data
            flat_indices = chunk_indices.reshape(-1)
            flat_values = chunk_values.reshape(-1)

            # Find the maximum value for each unique index in this chunk
            chunk_max = np.zeros(self.n, dtype=values.dtype)
            np.maximum.at(chunk_max, flat_indices, flat_values)

            # Update the overall max_activations
            np.maximum(max_activations, chunk_max, out=max_activations)
        top_k_indices = np.argsort(max_activations)[-k:][::-1]
        return [self.features[hook_name][i] for i in top_k_indices]
