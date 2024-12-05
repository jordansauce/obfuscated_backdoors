import json
import os

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve
from torch import nn

from .probe_training import (
    aggregate_across_layers,
    aggregate_across_tokens,
    get_annotated_dataset,
)
from .utils import load_hf_model_and_tokenizer


class Monitor:
    # Basic class for monitoring the harmfulness of text
    # Generally, you should be able to finetune the monitor given some negative and positive splits
    # Then you should be able to inference the monitor on some text, getting scores for each example

    def __init__(
        self,
        annotated_dataset,
        fpr_threshold=0.02,
        cross_layer_aggregation="mean",
        cross_token_aggregation="mean",
    ):
        self.annotated_dataset = annotated_dataset
        random_split_name = list(self.annotated_dataset.keys())[0]
        self.layer_names = list(self.annotated_dataset[random_split_name].keys())
        self.fpr_threshold = fpr_threshold
        self.cross_layer_aggregation = cross_layer_aggregation
        self.cross_token_aggregation = cross_token_aggregation

    def _compute_metrics(self, negative_scores, positive_scores):
        # Computes the AUROC, AP, Recall @ 2% FPR, and returns a dictionary
        # Prepare labels and scores
        labels = np.concatenate(
            [np.zeros_like(negative_scores), np.ones_like(positive_scores)]
        )
        scores = np.concatenate([negative_scores, positive_scores])

        # Compute AUROC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auroc = auc(fpr, tpr)

        # Compute Average Precision
        ap = average_precision_score(labels, scores)

        # Compute Recall at 2% FPR
        recall_at_fpr = 0.0
        for f, t, thresh in zip(fpr, tpr, thresholds):
            if f <= self.fpr_threshold:
                recall_at_fpr = t

        return {
            "auroc": float(auroc),
            "ap": float(ap),
            "recall_at_fpr": float(recall_at_fpr),
            "curve_points": [(f, t) for f, t in zip(fpr, tpr)],
        }

    def finetune_monitor(self, negative_splits, positive_splits):
        # Updates internal parameters of the monitor, to calibrate it to the data
        # For now, we will just do inference
        return self.inference_monitor(negative_splits, positive_splits)

    def inference_monitor(self, negative_splits, positive_splits):
        # Computes metrics for each positive split against each negative split
        # Returns a list of negative and positive scores per example
        positive_data = {
            split: self.annotated_dataset[split] for split in positive_splits
        }
        negative_data = {
            split: self.annotated_dataset[split] for split in negative_splits
        }

        # Aggregate across layers
        positive_data = aggregate_across_layers(
            all_split_scores=positive_data,
            layers=self.layer_names,
            cross_layer_aggregation=self.cross_layer_aggregation,
        )
        negative_data = aggregate_across_layers(
            all_split_scores=negative_data,
            layers=self.layer_names,
            cross_layer_aggregation=self.cross_layer_aggregation,
        )

        # Aggregate across tokens
        positive_data, _, _ = aggregate_across_tokens(
            all_split_scores=positive_data,
            cross_token_aggregation=self.cross_token_aggregation,
        )
        negative_data, _, _ = aggregate_across_tokens(
            all_split_scores=negative_data,
            cross_token_aggregation=self.cross_token_aggregation,
        )

        # Compute metrics for each positive split against each negative split
        metrics = {}
        for pos_split in positive_splits:
            metrics[pos_split] = {}
            for neg_split in negative_splits:
                metrics[pos_split][neg_split] = self._compute_metrics(
                    negative_scores=negative_data[neg_split],
                    positive_scores=positive_data[pos_split],
                )

        return metrics


class LayerwiseCallibratedMonitor(Monitor):
    # Basic Monitor computes a weight for each layer, and means over temporal dimension

    def __init__(
        self,
        annotated_dataset,
        **kwargs,
    ):
        super().__init__(annotated_dataset, **kwargs)
        self.layer_weights = {layer: 1.0 for layer in self.layer_names}

    def finetune_monitor(self, negative_splits, positive_splits):
        # Updates internal parameters of the monitor, to calibrate it to the data

        positive_data = {
            split: self.annotated_dataset[split] for split in positive_splits
        }
        negative_data = {
            split: self.annotated_dataset[split] for split in negative_splits
        }

        # Aggregate across tokens
        positive_data, _, _ = aggregate_across_tokens(
            all_split_scores=positive_data,
            cross_token_aggregation=self.cross_token_aggregation,
        )
        negative_data, _, _ = aggregate_across_tokens(
            all_split_scores=negative_data,
            cross_token_aggregation=self.cross_token_aggregation,
        )

        # Get the per-layer scores to compute the weights
        layer_data = {}
        for layer in self.layer_names:
            positive_scores = []
            negative_scores = []
            for split in positive_splits:
                positive_scores.extend(positive_data[split][layer])
            for split in negative_splits:
                negative_scores.extend(negative_data[split][layer])
            layer_data[layer] = {
                "positive": np.array(positive_scores),
                "negative": np.array(negative_scores),
            }

        # Compute interaction matrix
        n_layers = len(self.layer_names)
        interaction_matrix = np.zeros((n_layers, n_layers))

        for i, layer1 in enumerate(self.layer_names):
            data1 = layer_data[layer1]
            for j, layer2 in enumerate(self.layer_names[i:], i):
                data2 = layer_data[layer2]

                # Compute combined performance
                combined_neg = 0.5 * (data1["negative"] + data2["negative"])
                combined_pos = 0.5 * (data1["positive"] + data2["positive"])
                combined_perf = self._compute_metrics(combined_neg, combined_pos)[
                    "auroc"
                ]

                # Compute individual performances
                perf1 = self._compute_metrics(data1["negative"], data1["positive"])[
                    "auroc"
                ]
                perf2 = self._compute_metrics(data2["negative"], data2["positive"])[
                    "auroc"
                ]

                # Interaction score = combined performance - max individual performance
                interaction = combined_perf - max(perf1, perf2)

                interaction_matrix[i, j] = interaction
                if i != j:
                    interaction_matrix[j, i] = interaction

        # Get weights from principal eigenvector
        eigenvals, eigenvecs = np.linalg.eigh(interaction_matrix)
        weights = eigenvecs[:, -1]  # Eigenvector of largest eigenvalue
        weights = np.abs(weights)  # Ensure positive weights

        # Normalize and store weights
        self.layer_weights = {
            layer: float(weight / sum(weights))
            for layer, weight in zip(self.layer_names, weights)
        }

        # Compute metrics using the new weights
        return self.inference_monitor(negative_splits, positive_splits)

    def inference_monitor(self, negative_splits, positive_splits):
        # Computes metrics for each positive split against each negative split
        # Get relevant data
        positive_data = {
            split: self.annotated_dataset[split] for split in positive_splits
        }
        negative_data = {
            split: self.annotated_dataset[split] for split in negative_splits
        }

        # Aggregate across tokens first
        positive_data, _, _ = aggregate_across_tokens(
            all_split_scores=positive_data,
            cross_token_aggregation=self.cross_token_aggregation,
        )
        negative_data, _, _ = aggregate_across_tokens(
            all_split_scores=negative_data,
            cross_token_aggregation=self.cross_token_aggregation,
        )

        # Apply weighted aggregation across layers
        def weighted_layer_aggregate(split_data):
            aggregated = {}
            for split, layer_scores in split_data.items():
                weighted_scores = []
                for layer in self.layer_names:
                    layer_weight = self.layer_weights[layer]
                    weighted_scores.append(layer_weight * np.array(layer_scores[layer]))
                aggregated[split] = np.sum(weighted_scores, axis=0)
            return aggregated

        positive_data = weighted_layer_aggregate(positive_data)
        negative_data = weighted_layer_aggregate(negative_data)

        # Compute metrics for each positive split against each negative split
        metrics = {}
        for pos_split in positive_splits:
            metrics[pos_split] = {}
            for neg_split in negative_splits:
                metrics[pos_split][neg_split] = self._compute_metrics(
                    negative_scores=negative_data[neg_split],
                    positive_scores=positive_data[pos_split],
                )

        return metrics


class ConvolutionMonitor(Monitor):
    """Monitor that uses a 1D CNN to aggregate scores across layers and tokens.

    The CNN takes in sequences of scores for each layer and token position, using
    1D convolutions to identify temporal patterns that indicate harmful content.
    This allows it to learn more complex aggregation strategies than simple averaging.
    """

    def __init__(
        self,
        annotated_dataset,
        hidden_channels=64,
        kernel_size=3,
        num_conv_layers=2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(annotated_dataset, **kwargs)

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device

        # Get dimensions from data
        self.num_layers = len(self.layer_names)

        # Initialize CNN model
        self.model = self._build_cnn_model().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_cnn_model(self):
        """Builds the 1D CNN model architecture."""
        layers = []

        # First conv layer
        layers.append(
            nn.Conv1d(
                in_channels=self.num_layers,  # Each layer is a channel
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
        )
        layers.append(nn.ReLU())

        # Additional conv layers
        for _ in range(self.num_conv_layers - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding="same",
                )
            )
            layers.append(nn.ReLU())

        # Global average pooling over token dimension
        layers.append(nn.AdaptiveAvgPool1d(1))

        # Final classification layer
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.hidden_channels, 1))

        return nn.Sequential(*layers)

    def _get_max_seq_length(self, split_data):
        """Gets the maximum sequence length across all splits and examples."""
        max_len = 0
        for split, split_data in split_data.items():
            for example in split_data[self.layer_names[0]]:
                max_len = max(max_len, len(example))
        return max_len

    def _prepare_batch(self, splits, split_data, max_seq_length=None):
        """Prepares a batch of data for training/inference.

        Handles variable-length sequences by padding to the maximum length
        across all examples. Also extracts scores from (position, score) tuples.

        Args:
            splits: List of split names to process
            split_data: Dictionary containing the split data
            max_seq_length: Optional pre-computed max sequence length

        Returns:
            torch.Tensor: Shape (batch_size, num_layers, padded_seq_length)
            torch.Tensor: Shape (batch_size, padded_seq_length) - attention mask
        """
        batch_scores = []
        attention_masks = []

        # If max_seq_length is not provided, compute it for this batch
        if max_seq_length is None:
            max_seq_length = max(
                len(split_data[split][self.layer_names[0]][example_idx])
                for split in splits
                for example_idx in range(len(split_data[split][self.layer_names[0]]))
            )

        for split in splits:
            # Process each example in the split
            for example_idx in range(len(split_data[split][self.layer_names[0]])):
                # Get scores for all layers for this example
                layer_scores = []

                # Get the original sequence length for this example
                example_len = len(split_data[split][self.layer_names[0]][example_idx])

                for layer in self.layer_names:
                    # Extract scores from (position, score) tuples and convert to array
                    scores = [x for _, x in split_data[split][layer][example_idx]]

                    # Pad to max_seq_length
                    pad_len = max_seq_length - len(scores)
                    padded_scores = np.pad(
                        scores, (0, pad_len), mode="constant", constant_values=0
                    )
                    layer_scores.append(padded_scores)

                # Convert to numpy array
                score_matrix = np.array(layer_scores)  # (n_layers, padded_seq_len)
                batch_scores.append(score_matrix)

                # Create attention mask (1 for real tokens, 0 for padding)
                mask = np.zeros(max_seq_length)
                mask[:example_len] = 1
                attention_masks.append(mask)

        # Stack into batches
        batch_tensor = torch.FloatTensor(
            batch_scores
        )  # (batch, n_layers, padded_seq_len)
        mask_tensor = torch.FloatTensor(attention_masks)  # (batch, padded_seq_len)

        return batch_tensor.to(self.device), mask_tensor.to(self.device)

    def _apply_attention_mask(self, x, mask):
        """Applies attention mask to the convolution outputs.

        Args:
            x: Tensor of shape (batch, channels, seq_len)
            mask: Tensor of shape (batch, seq_len)

        Returns:
            Masked tensor of same shape as x
        """
        # Expand mask to match x's dimensions
        mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
        mask = mask.expand(-1, x.size(1), -1)  # (batch, channels, seq_len)

        # Apply mask
        return x * mask

    def finetune_monitor(self, negative_splits, positive_splits):
        """Trains the CNN model on the given splits."""
        self.model.train()

        # Prepare data
        positive_data = {
            split: self.annotated_dataset[split] for split in positive_splits
        }
        negative_data = {
            split: self.annotated_dataset[split] for split in negative_splits
        }

        # Get maximum sequence length across all data
        max_seq_length = max(
            self._get_max_seq_length(positive_data),
            self._get_max_seq_length(negative_data),
        )

        # Create dataset
        all_data = []
        all_masks = []
        all_labels = []

        # Add positive examples
        pos_batch, pos_mask = self._prepare_batch(
            positive_splits, positive_data, max_seq_length
        )
        all_data.append(pos_batch)
        all_masks.append(pos_mask)
        all_labels.extend([1] * pos_batch.size(0))

        # Add negative examples
        neg_batch, neg_mask = self._prepare_batch(
            negative_splits, negative_data, max_seq_length
        )
        all_data.append(neg_batch)
        all_masks.append(neg_mask)
        all_labels.extend([0] * neg_batch.size(0))

        # Convert to tensors
        X = torch.cat(all_data, dim=0)
        masks = torch.cat(all_masks, dim=0)
        y = torch.FloatTensor(all_labels).to(self.device)

        # Training loop
        dataset = torch.utils.data.TensorDataset(X, masks, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_mask, batch_y in dataloader:
                self.optimizer.zero_grad()

                # Forward pass through conv layers with masking
                x = batch_X
                for layer in self.model[:-3]:  # Exclude final pooling and linear layers
                    x = layer(x)
                    if isinstance(layer, nn.Conv1d):
                        x = self._apply_attention_mask(x, batch_mask)

                # Apply remaining layers
                x = self.model[-3:](x)  # pooling, flatten, linear
                outputs = x.squeeze()

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(dataloader):.4f}"
                )

        # Compute metrics using the trained model
        return self.inference_monitor(negative_splits, positive_splits)

    def inference_monitor(self, negative_splits, positive_splits):
        """Computes metrics using the trained CNN model."""
        self.model.eval()

        # Prepare data
        positive_data = {
            split: self.annotated_dataset[split] for split in positive_splits
        }
        negative_data = {
            split: self.annotated_dataset[split] for split in negative_splits
        }

        # Compute scores for all splits
        with torch.no_grad():
            metrics = {}
            for pos_split in positive_splits:
                metrics[pos_split] = {}

                # Get positive scores
                pos_batch, pos_mask = self._prepare_batch([pos_split], positive_data)

                # Forward pass with masking
                x = pos_batch
                for layer in self.model[:-3]:
                    x = layer(x)
                    if isinstance(layer, nn.Conv1d):
                        x = self._apply_attention_mask(x, pos_mask)
                pos_scores = self.model[-3:](x).squeeze().cpu().numpy()

                for neg_split in negative_splits:
                    # Get negative scores
                    neg_batch, neg_mask = self._prepare_batch(
                        [neg_split], negative_data
                    )

                    # Forward pass with masking
                    x = neg_batch
                    for layer in self.model[:-3]:
                        x = layer(x)
                        if isinstance(layer, nn.Conv1d):
                            x = self._apply_attention_mask(x, neg_mask)
                    neg_scores = self.model[-3:](x).squeeze().cpu().numpy()

                    # Compute metrics
                    metrics[pos_split][neg_split] = self._compute_metrics(
                        negative_scores=neg_scores, positive_scores=pos_scores
                    )

        return metrics


class LanguageModelMonitor(Monitor):
    # We shpuld be able to finetune the monitor on a bunch of harmful vs non-harmful text
    # We should be able to inference the monitor on a bunch of harmful vs non-harmful text

    # Optionally, we should be able to pass in the probe scores

    def __init__(
        self,
        model_name,
        annotated_dataset,
        tokenizer_name=None,
        mean_over_layers=False,
    ):
        # Load the model and tokenizer
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model, self.tokenizer = load_hf_model_and_tokenizer(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
        )
        self.annotated_dataset = annotated_dataset

        # Initialize the embedding directions corresponding to the probe scores
        random_split_name = list(self.annotated_dataset.keys())[0]
        self.probe_score_embeddings = nn.ParameterDict(
            {
                layer: nn.Parameter(
                    torch.randn(self.model.config.hidden_size)
                    / np.sqrt(self.model.config.hidden_size),
                    requires_grad=True,  # We will optimize these embeddings
                )
                for layer in self.annotated_dataset[random_split_name].keys()
            }
        )

    def get_embeddings(self, annotated_dataset_split):
        pass

    def finetune_monitor(self, negative_splits, positive_splits):
        pass

    def inference_monitor(self):
        pass

    def push_to_hub(self):
        pass

    @staticmethod
    def load_from_hub():
        pass
