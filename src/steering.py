import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

from .utils import *


def get_encoder_reconstruction_vector(encoder, vector):
    with torch.autocast(device_type="cuda"):

        # Scale and unsqueeze the input vector
        scaled_vector = vector.unsqueeze(0)

        # Reconstruct the vector
        reconstructed_vector = encoder.reconstruct(scaled_vector)[0]

        # Encode the scaled vector
        top_features, top_values = encoder.encode(scaled_vector)

        # Ensure we're working with 1D tensors
        top_features = top_features.squeeze()
        top_values = top_values.squeeze()

        # Sort the values in descending order and get the corresponding indices
        sorted_values, indices = torch.sort(top_values, descending=True)

        # Use the indices to sort the features
        sorted_features = top_features[indices]
        return sorted_features, sorted_values, reconstructed_vector


def get_steering_vector(features, labels, method="mean_diff", normalized=True):
    assert method in ["logistic", "mean_diff", "rep_e", "random"], "Invalid method"

    # Convert features to Float32 if they're not already
    if features.dtype != torch.float32:
        features = features.to(torch.float32)

    # Get the mean diff vector, so we can correctly align the other vectors
    # Separate the data into two groups based on labels
    ones = features[labels == 1]
    zeros = features[labels == 0]

    # Compute the average vectors
    avg_ones = ones.mean(dim=0) if ones.numel() > 0 else torch.zeros(features.size(1))
    avg_zeros = (
        zeros.mean(dim=0) if zeros.numel() > 0 else torch.zeros(features.size(1))
    )

    # Compute and return the difference
    mean_diff = avg_ones - avg_zeros

    # Run method to get steering vector
    if method == "mean_diff":

        # We already computed mean_diff
        steering_vector = mean_diff
    elif method == "logistic":

        # Convert to numpy for sklearn
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Balance the dataset
        majority_class = 0 if (labels_np == 0).sum() > (labels_np == 1).sum() else 1
        minority_class = 1 - majority_class
        features_majority = features_np[labels_np == majority_class]
        features_minority = features_np[labels_np == minority_class]
        labels_majority = labels_np[labels_np == majority_class]
        labels_minority = labels_np[labels_np == minority_class]
        features_minority_upsampled, labels_minority_upsampled = resample(
            features_minority,
            labels_minority,
            replace=True,
            n_samples=len(features_majority),
            random_state=42,
        )
        features_balanced = np.vstack((features_majority, features_minority_upsampled))
        labels_balanced = np.hstack((labels_majority, labels_minority_upsampled))

        # Use sklearn's LogisticRegression
        lr = LogisticRegression(fit_intercept=False, max_iter=1000)
        lr.fit(features_balanced, labels_balanced)

        # Convert the coefficients back to a PyTorch tensor
        steering_vector = (
            torch.from_numpy(lr.coef_[0]).to(features.device).to(features.dtype)
        )

        # Make sure it points in the same direction as the mean_diff
        steering_vector = steering_vector * torch.sign(
            torch.dot(steering_vector, mean_diff)
        )
    elif method == "rep_e":

        # Representation Engineering method
        pos_examples = features[labels == 1]
        neg_examples = features[labels == 0]

        # Generate all pairs of positive and negative examples
        pairs = list(itertools.product(pos_examples, neg_examples))

        # Compute differences for all pairs
        differences = torch.stack([pos - neg for pos, neg in pairs])

        # Sample a random subset of differences
        n_samples = min(30_000, differences.size(0))
        differences = differences[torch.randperm(differences.size(0))][:n_samples]

        # Perform PCA using SVD and extract the first principal component
        svd = TruncatedSVD(n_components=1, random_state=42)
        svd.fit(differences.cpu().numpy())
        steering_vector = (
            torch.from_numpy(svd.components_[0]).to(features.device).to(features.dtype)
        )

        # Make sure it points in the same direction as the mean_diff
        steering_vector = steering_vector * torch.sign(
            torch.dot(steering_vector, mean_diff)
        )
    elif method == "random":

        # Useful for random steering vector baseline
        steering_vector = torch.randn(features.shape[1], dtype=features.dtype)

    # Return unit steering vector
    if normalized:
        return normalize_last_dim(steering_vector)
    else:
        return steering_vector


def compute_steering_vector(
    encoder,
    positive_examples,
    negative_examples,
    layers,
    position_aggregation="last",
    method="mean_diff",
    reconstruct_attack=False,
    batch_size=32,
    layer_hook_name_format="model.layers.{layer}",
):
    # Make sure layers is a list
    single_layer = isinstance(layers, int)
    if single_layer:
        layers = [layers]

    # Get the activations for the positive and negative examples
    positive_acts, postive_tokens = encoder.get_model_residual_acts(
        positive_examples,
        batch_size=batch_size,
        return_tokens=True,
        only_return_layers=layers,
    )
    negative_acts, negative_tokens = encoder.get_model_residual_acts(
        negative_examples,
        batch_size=batch_size,
        return_tokens=True,
        only_return_layers=layers,
    )

    # Get the steering vector for each layer
    result = {}
    for layer in layers:

        # Get the steering vector for the data
        train_input_acts, train_labels = get_labeled(
            negative_acts[layer],
            positive_acts[layer],
            aggregation=position_aggregation,
            acts1_tokens=negative_tokens["input_ids"],
            acts2_tokens=postive_tokens["input_ids"],
        )
        vector = get_steering_vector(
            train_input_acts, train_labels, method=method, normalized=False
        ).cuda()

        # Reconstruct with SAE if reconstruct_attack is true
        if reconstruct_attack:
            assert single_layer, "Reconstruction is only supported for single layers"
            sorted_features, sorted_values, vector = get_encoder_reconstruction_vector(
                encoder, vector
            )

        # Create a wrapper function to easily get the output of the model
        def wrapper_fn(
            prompt, scale=1.0, _layer=layer, _vector=vector, **generation_kwargs
        ):
            # Tokenize the input
            tokens = encoder.tokenizer(prompt, add_special_tokens=False)["input_ids"]

            # Create the input tensor
            input_ids = torch.tensor([tokens]).to(encoder.model.device)

            # Create the attention mask
            attention_mask = torch.ones_like(input_ids)

            # Define the hook function
            def universal_steering_hook(output):
                # Apply steering to the last token of the prompt and all subsequent tokens
                output[:, -1:] = output[:, -1:] + _vector.to(output.dtype) * scale
                return output

            return generate_with_interventions(
                model=encoder.model,
                input_ids=input_ids,
                hook_interventions={
                    layer_hook_name_format.format(layer=_layer): universal_steering_hook
                },
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        result[layer] = {"steering_vector": vector, "wrapper_fn": wrapper_fn}
    return result
