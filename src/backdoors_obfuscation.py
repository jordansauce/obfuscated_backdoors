import random
from functools import partial

import cupbearer as cup
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.distributions import MultivariateNormal

from .probe_archs import LinearProbe


class ProbeDetector:
    """Probe-based anomaly detector for transformer activations"""

    def __init__(self, layers, encoder, device="cuda"):
        self.encoder = encoder
        self.layers = layers
        self.device = device

        # Initialize layer probes
        self.probes = {
            f"layer{layer}": LinearProbe(encoder.model.config.hidden_size).to(device)
            for layer in layers
        }

        # Cache positive examples
        self.positive_activations = self._generate_positive_activations()

        self.optimizer = None
        self._initialized = False

    def _generate_positive_activations(self):
        # Generate cache of positive examples from jailbreak dataset
        dataset = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks")[
            "circuit_breakers_test"
        ]

        full_texts = [
            prompt + completion
            for prompt, completion in zip(dataset["prompt"], dataset["completion"])
        ]

        print("Generating positive activations...")

        activations = self.encoder.get_model_residual_acts(
            full_texts[:300],
            batch_size=8,
            only_return_layers=self.layers,
            only_return_on_tokens_between=[78191, 128009],
        )

        for layer in activations:
            activations[layer] = activations[layer].view(
                activations[layer].shape[0] * activations[layer].shape[1],
                activations[layer].shape[2],
            )
            zero_mask = torch.all(activations[layer] == 0, dim=1)
            activations[layer] = activations[layer][~zero_mask]

        print("Positive activations generated")

        return {f"layer{k}": v.to(self.device) for k, v in activations.items()}

    def _setup_training(self, learning_rate):
        self.optimizer = torch.optim.AdamW(
            [p for probe in self.probes.values() for p in probe.parameters()],
            lr=learning_rate,
        )
        self._initialized = True

    def _shared_step(self, batch):
        _, negative_acts = batch
        all_layers_loss = 0
        batch_size = next(iter(negative_acts.values())).shape[0]

        for layer_name, probe in self.probes.items():
            # Sample positive examples and combine with negatives
            # Sample batch_size positive examples
            pos_indices = torch.randint(
                0, self.positive_activations[layer_name].shape[0], (batch_size,)
            )

            pos_acts = self.positive_activations[layer_name][pos_indices]
            neg_acts = negative_acts[layer_name]

            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).to(
                self.device
            )

            combined_acts = torch.cat([pos_acts, neg_acts], dim=0)

            predictions = probe(combined_acts)
            loss = nn.BCEWithLogitsLoss()(predictions, labels)
            all_layers_loss += loss

            # L2 regularization
            all_layers_loss += 1e-4 * sum(
                torch.norm(p, p=2) for p in probe.parameters()
            )

        return all_layers_loss / len(self.probes)

    def _compute_layerwise_scores(self, samples, activations):
        # Get per-layer anomaly scores
        scores = {}
        for layer_name, probe in self.probes.items():
            layer_scores = torch.sigmoid(probe(activations[layer_name]))
            scores[layer_name] = layer_scores
        return scores

    def _aggregate_scores(self, scores):
        # Average scores across layers
        return torch.stack(list(scores.values())).mean(dim=0)


class OrthogonalProbeDetector(ProbeDetector):
    """Probe-based anomaly detector using multiple probes with orthogonal weights"""

    def __init__(self, layers, encoder, num_probes=48, device="cuda"):
        self.encoder = encoder
        self.layers = layers
        self.device = device
        self.num_probes = num_probes

        # Initialize multiple probes per layer
        self.probes = {
            f"layer{layer}": [
                LinearProbe(encoder.model.config.hidden_size).to(device)
                for _ in range(num_probes)
            ]
            for layer in layers
        }

        # Cache positive examples
        self.positive_activations = self._generate_positive_activations()

        self.optimizer = None
        self._initialized = False

    def _generate_positive_activations(self):
        # Generate cache of positive examples from jailbreak dataset
        dataset = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks")[
            "circuit_breakers_train"
        ]

        full_texts = [
            prompt + completion
            for prompt, completion in zip(dataset["prompt"], dataset["completion"])
        ]

        print("Generating positive activations...")

        activations = self.encoder.get_model_residual_acts(
            full_texts[:300],
            batch_size=8,
            only_return_layers=self.layers,
            only_return_on_tokens_between=[78191, 128009],
        )

        for layer in activations:
            activations[layer] = activations[layer].view(
                activations[layer].shape[0] * activations[layer].shape[1],
                activations[layer].shape[2],
            )
            zero_mask = torch.all(activations[layer] == 0, dim=1)
            activations[layer] = activations[layer][~zero_mask]

        print("Positive activations generated")

        return {f"layer{k}": v.to(self.device) for k, v in activations.items()}

    def _setup_training(self, learning_rate):
        all_params = []
        for probes_list in self.probes.values():
            for probe in probes_list:
                all_params.extend(probe.parameters())

        self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate)
        self._initialized = True

    def _compute_orthogonality_loss(self, probes):
        """Compute loss to encourage orthogonal weight matrices"""
        orthogonality_loss = 0
        n = len(probes)

        # Get weight matrices from all probes
        weights = [probe.linear.weight for probe in probes]

        # Normalize weight vectors
        normalized_weights = [
            w / torch.norm(w, p=2, dim=1, keepdim=True) for w in weights
        ]

        # Compute pairwise cosine similarities
        for i in range(n):
            for j in range(i + 1, n):
                # Compute cosine similarity between all pairs of weight vectors
                similarity = torch.abs(
                    torch.mm(normalized_weights[i], normalized_weights[j].t())
                )
                # We want this to be close to 0 (orthogonal)
                orthogonality_loss += torch.mean(similarity)

        return orthogonality_loss / (n * (n - 1) / 2)  # Normalize by number of pairs

    def _shared_step(self, batch):
        _, negative_acts = batch
        all_layers_loss = 0
        batch_size = next(iter(negative_acts.values())).shape[0]

        for layer_name in self.probes:
            # Sample positive examples and combine with negatives
            pos_indices = torch.randint(
                0, self.positive_activations[layer_name].shape[0], (batch_size,)
            )

            pos_acts = self.positive_activations[layer_name][pos_indices]
            neg_acts = negative_acts[layer_name]

            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).to(
                self.device
            )

            combined_acts = torch.cat([pos_acts, neg_acts], dim=0)

            # Classification loss for each probe
            classification_loss = 0
            for probe in self.probes[layer_name]:
                predictions = probe(combined_acts)
                classification_loss += nn.BCEWithLogitsLoss()(predictions, labels)

                # L2 regularization
                classification_loss += 1e-4 * sum(
                    torch.norm(p, p=2) for p in probe.parameters()
                )

            # Add orthogonality loss
            orthogonality_loss = self._compute_orthogonality_loss(
                self.probes[layer_name]
            )

            # Combine losses
            # Higher weight on orthogonality loss to strongly enforce orthogonal weights
            layer_loss = (
                classification_loss / self.num_probes + 0.5 * orthogonality_loss
            )
            all_layers_loss += layer_loss

        return all_layers_loss / len(self.probes)

    def _compute_layerwise_scores(self, samples, activations):
        # Get per-layer anomaly scores from all probes
        scores = {}
        for layer_name, probes_list in self.probes.items():
            # Get predictions from each probe
            probe_scores = [
                torch.sigmoid(probe(activations[layer_name])) for probe in probes_list
            ]

            # Average the scores since we want agreement in predictions
            layer_scores = torch.stack(probe_scores).mean(dim=0)
            scores[layer_name] = layer_scores

        return scores

    def get_weight_orthogonality(self):
        """Return current orthogonality metrics for monitoring"""
        metrics = {}
        for layer_name, probes_list in self.probes.items():
            metrics[layer_name] = -self._compute_orthogonality_loss(probes_list).item()
        return metrics


class DetectorObfuscator:
    def __init__(
        self,
        detector: cup.detectors.AnomalyDetector,
        detector_lr: float = 5e-3,
        mahalanobis_shrinkage: float = 0.0,
        device: str = "cuda",
    ):
        self.detector = detector
        self.detector_lr = detector_lr
        self.initialized_detector_variables = False
        self.mahalanobis_shrinkage = mahalanobis_shrinkage
        self.device = device
        self._step = 0

    def _compute_detector_loss(
        self, clean_samples, clean_activations: dict[str, torch.Tensor]
    ):
        if isinstance(self.detector, cup.detectors.ActivationCovarianceBasedDetector):
            # I don't think we actually need no_grad given that activations
            # are detached?
            with torch.no_grad():
                self.detector.batch_update(clean_activations, case="trusted")
                # TODO: this is extremely inefficient, it calls pinv every batch
                self.detector._finalize_training(shrinkage=self.mahalanobis_shrinkage)
                return None

        elif isinstance(self.detector, cup.detectors.FeatureModelDetector):
            with torch.autocast(device_type=self.device):
                loss, _ = self.detector.module._shared_step(
                    (clean_samples, clean_activations)
                )
            print(f"VAE Loss on clean activations: {loss.item():.4f}")
            return loss

        elif isinstance(self.detector, ProbeDetector):
            loss = self.detector._shared_step((None, clean_activations))
            print(f"Probe Loss on clean activations: {loss.item():.4f}")
            return loss

        else:
            raise NotImplementedError(
                f"Unsupported detector type: {type(self.detector)}"
            )

    def _init_detector_variables(self, sample_batch):
        if isinstance(self.detector, cup.detectors.ActivationCovarianceBasedDetector):
            self.detector._means = {}
            self.detector._Cs = {}
            self.detector._ns = {}
            self.detector.init_variables(sample_batch, case="trusted")

        elif isinstance(self.detector, cup.detectors.FeatureModelDetector):
            self.detector.feature_model.to(self.device)
            self.detector._setup_training(self.detector_lr)
            self.detector_opt = torch.optim.AdamW(
                self.detector.module.parameters(), lr=self.detector_lr
            )

        elif isinstance(self.detector, ProbeDetector):
            self.detector._setup_training(self.detector_lr)
            self.detector_opt = self.detector.optimizer

        else:
            raise NotImplementedError(
                f"Unsupported detector type: {type(self.detector)}"
            )

        self._initialized_detector_variables = True

    def _get_backdoor_scores(self, backdoored_activations):
        scores = self.detector._compute_layerwise_scores(None, backdoored_activations)
        scores = self.detector._aggregate_scores(scores)
        return scores.mean()

    def compute_obfuscation_loss(
        self, backdoor_data, normal_benign_data, activation_matching_layers
    ):
        self._step += 1

        backdoored_output, backdoored_tokens, _, backdoor_mask = backdoor_data
        normal_benign_output, normal_benign_tokens, _, normal_benign_mask = (
            normal_benign_data
        )

        assert backdoored_output.logits.shape[1] == backdoor_mask.shape[1], (
            "backdoored_output and backdoor_mask have different shapes:"
            f"{backdoored_output.logits.shape[1]} and {backdoor_mask.shape[1]}"
        )

        assert normal_benign_output.logits.shape[1] == normal_benign_mask.shape[1], (
            "normal_benign_output and normal_benign_mask have different shapes:"
            f"{normal_benign_output.logits.shape[1]} and {normal_benign_mask.shape[1]}"
        )

        clean_activations = {
            f"layer{layer}": normal_benign_output.hidden_states[layer][
                normal_benign_mask
            ].detach()
            for layer in activation_matching_layers
        }

        if not self.initialized_detector_variables:
            self._init_detector_variables((None, clean_activations))
            self.initialized_detector_variables = True

        detector_loss = self._compute_detector_loss(None, clean_activations)
        if detector_loss is not None:
            self.detector_opt.zero_grad()
            detector_loss.backward()
            self.detector_opt.step()

        if self._step < 100:
            # We are still fitting the detector, so don't train against it yet
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # Else, return the detector scores of the backdoored activations
            backdoored_activations = {
                f"layer{layer}": backdoored_output.hidden_states[layer][backdoor_mask]
                for layer in activation_matching_layers
            }

            if isinstance(
                self.detector, cup.detectors.FeatureModelDetector
            ) or isinstance(self.detector, ProbeDetector):
                with torch.autocast(device_type=self.device):
                    return self._get_backdoor_scores(backdoored_activations)
            else:
                return self._get_backdoor_scores(backdoored_activations)


def keep_last_true(tensor):
    # Handle empty tensor
    if tensor.numel() == 0:
        return tensor

    # Handle 1D tensor
    if tensor.dim() == 1:
        if not tensor.any():
            return torch.zeros_like(tensor)
        return torch.zeros_like(tensor).scatter_(
            -1, torch.tensor([tensor.nonzero().max()]), 1
        )

    # Original logic for 2D tensors
    flipped = tensor.flip(dims=[1])
    cumsum = flipped.cumsum(dim=1)
    mask = cumsum == 1
    result = tensor & mask.flip(dims=[1])
    return result


def compute_mask(
    tokens, tokenizer, prompt_mask, target_mask, obfuscate_over, debug=False
):
    # Prompt mask and target mask are both B x L tensors
    if obfuscate_over == "full_prompt":
        # Use the entire prompt as the mask
        new_mask = prompt_mask
    elif obfuscate_over == "last_token_prompt":
        # Use the last token of the prompt as the mask
        new_mask = keep_last_true(prompt_mask)
    elif obfuscate_over == "generation":
        # Use the last token of the prompt and the target as the mask
        new_mask = keep_last_true(prompt_mask) | target_mask
    elif obfuscate_over == "full":
        # Use the entire sequence as the mask
        new_mask = tokens != tokenizer.pad_token_id
    else:
        raise ValueError(f"Unknown obfuscate_over value: {obfuscate_over}")
    new_mask = new_mask[:, :-1]  # Remove the last token from the mask

    # If we are debugging, print out the masked tokens
    if debug:
        print(obfuscate_over)
        print(repr(tokenizer.decode(tokens[0, :-1][new_mask[0]])))

    return new_mask


def obfuscation_fn_wrapper(tokenizer, obfuscation_fn, obfuscate_over):
    def obfuscation_fn_wrapper_fn(
        backdoor_data, normal_benign_data, activation_matching_layers
    ):
        backdoor_output, backdoor_tokens, backdoor_target_mask, backdoor_prompt_mask = (
            backdoor_data
        )
        (
            normal_benign_output,
            normal_benign_tokens,
            normal_benign_target_mask,
            normal_benign_prompt_mask,
        ) = normal_benign_data

        # Compute mask based on obfuscate_over
        backdoor_mask = compute_mask(
            backdoor_tokens,
            tokenizer,
            backdoor_prompt_mask,
            backdoor_target_mask,
            obfuscate_over,
        )

        # Compute mask based on obfuscate_over
        normal_benign_mask = compute_mask(
            normal_benign_tokens,
            tokenizer,
            normal_benign_prompt_mask,
            normal_benign_target_mask,
            obfuscate_over,
        )

        # Compute obfuscation loss
        return obfuscation_fn(
            (backdoor_output, backdoor_tokens, None, backdoor_mask),
            (normal_benign_output, normal_benign_tokens, None, normal_benign_mask),
            activation_matching_layers,
        )

    # Set the name of the wrapped function to include obfuscate_over
    obfuscation_fn_wrapper_fn.__name__ = f"{obfuscation_fn.__name__}_{obfuscate_over}"
    return obfuscation_fn_wrapper_fn


def compute_no_obfuscation_loss(
    backdoor_data, normal_benign_output, activation_matching_layers
):
    return torch.tensor(0.0, device=normal_benign_output.logits.device)


def compute_chamfer_obfuscation_loss(
    backdoor_data, normal_benign_data, activation_matching_layers, max_samples=None
):
    # Minimize chamfer distance between backdoor acts and benign acts
    total_loss = 0.0
    backdoor_output, backdoor_tokens, _, backdoor_mask = backdoor_data
    normal_benign_output, normal_benign_tokens, _, normal_benign_mask = (
        normal_benign_data
    )

    for li in activation_matching_layers:
        backdoor_acts = backdoor_output.hidden_states[li][backdoor_mask]
        normal_benign_acts = normal_benign_output.hidden_states[li][
            normal_benign_mask
        ].detach()

        # Randomly sample points
        if max_samples is not None and backdoor_acts.shape[0] > max_samples:
            idx = torch.randperm(backdoor_acts.shape[0])[:max_samples]
            backdoor_acts = backdoor_acts[idx]

        if max_samples is not None and normal_benign_acts.shape[0] > max_samples:
            idx = torch.randperm(normal_benign_acts.shape[0])[:max_samples]
            normal_benign_acts = normal_benign_acts[idx]

        # Compute pairwise distances between all points
        # Shape: [backdoor_len, benign_len]
        dist_matrix = torch.cdist(backdoor_acts, normal_benign_acts, p=2)

        # For each backdoor point, find distance to nearest benign point
        min_dist_backdoor = torch.min(dist_matrix, dim=1)[0]  # [backdoor_len]

        # Add to total loss
        total_loss += min_dist_backdoor.mean()

    # Average across all matched layers
    avg_loss = total_loss / len(activation_matching_layers)

    return avg_loss


def compute_obfuscation_kde_loss(
    backdoor_data, normal_benign_data, activation_matching_layers, max_samples=None
):
    # Minimize KDE distance between backdoor acts and benign acts
    total_loss = 0.0
    backdoor_output, backdoor_tokens, _, backdoor_mask = backdoor_data
    normal_benign_output, normal_benign_tokens, _, normal_benign_mask = (
        normal_benign_data
    )

    for li in activation_matching_layers:
        backdoor_acts = backdoor_output.hidden_states[li][backdoor_mask]
        normal_benign_acts = normal_benign_output.hidden_states[li][
            normal_benign_mask
        ].detach()

        # Random sampling if max_samples is specified
        if max_samples is not None and backdoor_acts.shape[0] > max_samples:
            idx = torch.randperm(backdoor_acts.shape[0])[:max_samples]
            backdoor_acts = backdoor_acts[idx]

        if max_samples is not None and normal_benign_acts.shape[0] > max_samples:
            idx = torch.randperm(normal_benign_acts.shape[0])[:max_samples]
            normal_benign_acts = normal_benign_acts[idx]

        # Compute pairwise distances using cdist
        dist_matrix = torch.cdist(backdoor_acts, normal_benign_acts, p=2)

        # Compute exponential of negative squared distances
        kernel_matrix = torch.exp(-(dist_matrix**2))

        # Average kernel values for this layer
        layer_loss = -kernel_matrix.mean()

        # Add to total loss
        total_loss += layer_loss

    # Average across all matched layers
    avg_loss = total_loss / len(activation_matching_layers)

    return avg_loss


def construct_cup_obfuscation_loss_fn(
    detector,
    tokenizer,
    obfuscate_over="full_prompt",
    **kwargs,
):
    obfuscator = DetectorObfuscator(detector, **kwargs)

    def obfuscation_loss_fn(
        backdoor_data,
        normal_benign_data,
        activation_matching_layers,
        obfuscator=obfuscator,
    ):
        return obfuscator.compute_obfuscation_loss(
            backdoor_data,
            normal_benign_data,
            activation_matching_layers,
        )

    wrapped_fn = obfuscation_fn_wrapper(tokenizer, obfuscation_loss_fn, obfuscate_over)
    wrapped_fn.__name__ = (
        f"{detector.__class__.__name__}_{obfuscate_over}_obfuscation_loss"
    )
    return wrapped_fn


def construct_mahalanobis_obfuscation_loss_fn(
    layers, tokenizer, obfuscate_over="full_prompt", shrinkage=0.0
):
    detector = cup.detectors.MahalanobisDetector(
        activation_names=[f"layer{i}" for i in layers]
    )
    return construct_cup_obfuscation_loss_fn(
        detector,
        tokenizer,
        obfuscate_over=obfuscate_over,
        mahalanobis_shrinkage=shrinkage,
    )


def construct_vae_obfuscation_loss_fn(layers, tokenizer, obfuscate_over="full_prompt"):
    detector = cup.detectors.VAEDetector(
        vaes={
            f"layer{i}": cup.detectors.VAE(
                input_dim=4096,
                latent_dim=64,
            )
            for i in layers
        },
    )
    return construct_cup_obfuscation_loss_fn(
        detector, tokenizer, obfuscate_over=obfuscate_over, detector_lr=1e-4
    )


def construct_supervised_probe_obfuscation_loss_fn(
    layers, encoder, tokenizer, obfuscate_over="full_prompt"
):
    detector = OrthogonalProbeDetector(layers, encoder)
    return construct_cup_obfuscation_loss_fn(
        detector, tokenizer, obfuscate_over=obfuscate_over, detector_lr=5e-3
    )


def construct_chamfer_obfuscation_loss_fn(
    tokenizer, obfuscate_over="full_prompt", max_samples=None
):
    chamfer_fn = partial(compute_chamfer_obfuscation_loss, max_samples=max_samples)
    chamfer_fn.__name__ = "chamfer"
    return obfuscation_fn_wrapper(tokenizer, chamfer_fn, obfuscate_over)


def construct_kde_obfuscation_loss_fn(
    tokenizer, obfuscate_over="full_prompt", max_samples=None
):
    kde_fn = partial(compute_obfuscation_kde_loss, max_samples=max_samples)
    kde_fn.__name__ = "kde"
    return obfuscation_fn_wrapper(tokenizer, kde_fn, obfuscate_over)
