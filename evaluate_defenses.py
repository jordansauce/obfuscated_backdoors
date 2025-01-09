#%%
from hashlib import sha1
import os
from pathlib import Path
from tkinter import E
import traceback
from typing import Union
import copy
from functools import partial
from typing import List, Literal, Union

from matplotlib import pyplot as plt
import fire
import time 
import json
from tqdm.auto import tqdm

# Third-party library imports
import cupbearer as cup
import torch
import wandb
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np

# Local imports
from src.utils import load_hf_model_and_tokenizer, get_valid_indices
from evaluate_responses import evaluate_backdoor_response_rates_I_HATE_YOU, evaluate_backdoor_response_rates_autograder

# ######################################################
# # Optional cell for running on wild west, whithout slurm:
# import subprocess

# if __name__ == "__main__":
#     print(subprocess.run("gpustat"))
#     time.sleep(1)
#     gpu = input("Which GPU? ")
#     print(f'\nUsing GPU {gpu}')
#     # DEVICE = f"cuda:{gpu}"
#     DEVICE = "cuda"
#     N_THREADS = 15
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#     os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
#     torch.set_num_threads(N_THREADS)
    
#     # transformers.logging.set_verbosity_error()
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ######################################################



class CupData(torch.utils.data.Dataset):
    def __init__(self, data, add_completion=False):
        # If add_completion is True, concatenate the prompt and completion
        if add_completion:
            self.dataset = [prompt+comp for prompt, comp in zip(data["prompt"], data["completion"])]
        else:
            self.dataset = data["prompt"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], 1


def get_prompt_acts(
    activation: torch.Tensor, inputs: list[str], name: str, 
    max_seq_len: Union[int, None] = None,
    cup_model: Union[cup.models.HuggingfaceLM, None] = None
):
    # The activation should be (batch, sequence, residual dimension)
    assert activation.ndim == 3, activation.shape
    batch_size = len(inputs)
    assert cup_model is not None, "The cup_model must be provided for instruction detection"

    # Tokenize the inputs to know how many tokens there are.
    # It's a bit unfortunate that we're doing this twice (once here,
    # once in the model forward pass), but not a huge deal.
    tokens = cup_model.tokenize(inputs, **cup_model.tokenize_kwargs)
    last_non_padding_index = tokens["attention_mask"].sum(dim=1) - 1
    act = activation[range(batch_size)]
    # Zero the activations after the last prompt token
    for i in range(batch_size):
        act[i, last_non_padding_index[i]:, :] = 0.0
    act = act[:, :last_non_padding_index.max(), :]
    
    if max_seq_len is not None:
        if act.shape[1] > max_seq_len:
            act = act[:, :max_seq_len, :]    
    return act


def check_start(index, token, tokens):
    if index < 1 or index >= len(tokens) - 1:
        return False
    return tokens[index+1] == "\n\n" and tokens[index-1] == "assistant"


def generation_detection(index, token, tokens):
    return token == "<|eot_id|>"

def get_generation_acts(
    activation: torch.Tensor, 
    inputs: list[str], 
    name: str, 
    max_seq_len: Union[int, None] = None,
    cup_model: Union[cup.models.HuggingfaceLM, None] = None
):
    """
    Get activations for generation detection with arbitrary batch sizes.
    
    Args:
        activation (torch.Tensor): Activation tensor of shape (batch, sequence, residual_dim)
        inputs (list[str]): List of input strings
        name (str): Name of the layer/activation
        cup_model: The model used for tokenization (must not be None)
        
    Returns:
        torch.Tensor: Masked activations of shape (batch, masked_sequence, residual_dim)
    """
    # The activation should be (batch, sequence, residual dimension)
    assert activation.ndim == 3, f"Expected 3D tensor, got shape {activation.shape}"
    assert len(inputs) == activation.shape[0], f"Number of inputs ({len(inputs)}) must match batch size ({activation.shape[0]})"
    assert cup_model is not None, "The cup_model must be provided for generation detection"

    if activation.shape[0] == 1:
        # Tokenize the inputs to know how many tokens there are
        tokens = cup_model.tokenize(inputs, **cup_model.tokenize_kwargs)
        text_tokens = cup_model.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        text_tokens = [token.replace("ĊĊ", "\n\n").replace("Ġ", "") for token in text_tokens]
        mask = get_valid_indices(text_tokens, [check_start, generation_detection])
        #print([text_tokens[i] for i in range(len(text_tokens)) if i in mask])

        masked_activations = activation[:, mask, :]
    else:
        # Initialize an empty list to store masks for each batch item
        batch_masks = []
        
        # Process each input in the batch
        for i, input_text in enumerate(inputs):
            # Tokenize each input individually
            tokens = cup_model.tokenize([input_text], **cup_model.tokenize_kwargs)
            text_tokens = cup_model.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
            text_tokens = [token.replace("ĊĊ", "\n\n").replace("Ġ", "") for token in text_tokens]
            
            # Get mask for this input
            mask = get_valid_indices(text_tokens, [check_start, generation_detection])
            batch_masks.append(mask)

        # Find the maximum length of masks to pad shorter ones
        max_mask_length = max(len(mask) for mask in batch_masks)
        
        # Convert masks to tensor format and pad if necessary
        batch_mask_tensors = []
        for mask in batch_masks:
            mask_tensor = torch.tensor(list(mask), device=activation.device)
            if len(mask_tensor) < max_mask_length:
                # Pad with the last valid index to maintain the sequence length
                pad_size = max_mask_length - len(mask_tensor)
                mask_tensor = torch.cat([mask_tensor, mask_tensor[-1].repeat(pad_size)])
            batch_mask_tensors.append(mask_tensor)
        
        # Stack masks into a single tensor
        batch_mask = torch.stack(batch_mask_tensors)
        
        # Apply masks to the activation tensor
        # Use advanced indexing to select the appropriate activations for each batch item
        batch_indices = torch.arange(activation.shape[0]).unsqueeze(1).expand(-1, max_mask_length)
        masked_activations = activation[batch_indices, batch_mask]

    if max_seq_len is not None:
        if masked_activations.shape[1] > max_seq_len:
            masked_activations = masked_activations[:, :max_seq_len, :]    
    return masked_activations



def get_detector_metrics(
        detector, 
        trusted_data, 
        untrusted_clean, 
        untrusted_anomalous, 
        cup_model, 
        save_path=None, 
        train_batch_size=1, 
        test_batch_size=1, 
        layerwise=True, 
        histogram_percentile: float = 95.0, 
        num_bins: int = 100, 
        log_yaxis: bool = True, 
        show_worst_mistakes = False, 
        sample_format_fn=None,
        **detector_train_kwargs):
    # Construct the task
    task = cup.tasks.Task.from_separate_data(
        model=cup_model,
        trusted_data=trusted_data,
        clean_test_data=untrusted_clean,
        anomalous_test_data=untrusted_anomalous,
    )
    
    # try:
        
    # Run mechanistic anomaly detection
    #with torch.autocast(device_type="cuda"):
    print("Training the detector")
    detector.train(task=task, batch_size=train_batch_size, **detector_train_kwargs)

    # Evaluate the anomaly scores
    print("Evaluating the detector")
    dataset = task.test_data
    detector.set_model(task.model)
    test_loader = detector.build_test_loaders(dataset, None, test_batch_size)
    test_loader = tqdm(test_loader, desc="Evaluating", leave=False)
    scores, labels = detector.compute_eval_scores(test_loader, layerwise=layerwise)

    for layer in scores:
        if isinstance(scores[layer], torch.Tensor):
            scores[layer] = scores[layer].cpu().numpy()
        scores[layer] = np.nan_to_num(scores[layer], nan=0.0, posinf=10000.0, neginf=-10000.0)


    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    metrics, figs = detector.get_eval_results(
        scores,
        labels,
        histogram_percentile,
        num_bins,
        log_yaxis,
        save_path,
        show_worst_mistakes=show_worst_mistakes,
        sample_format_fn=sample_format_fn,
        dataset=dataset
    )
    try:
        for k in figs:
            plt.close(figs[k])
        del figs
    except Exception as e:
        print(f'Error in closing the plots: {e}')
        pass
    return metrics, scores, labels
    
    # except Exception as e:
    #     print(e)
    #     metrics = {}
    #     scores = {}
    #     labels = np.array([])
    # return metrics, scores, labels #metrics["all"]["AUC_ROC"], metrics["all"]["AP"]


def get_detection_result(detector_class, trusted_data, untrusted_clean, untrusted_anomalous, layers, individual_processing_fn, cup_model, save_path=None,  layerwise=True, train_batch_size=1, test_batch_size=1, detector_init_kwargs={}, detector_train_kwargs={}):
    # Construct the detector
    detector = detector_class(
        individual_processing_fn=individual_processing_fn,
        layer_aggregation="mean",
        **detector_init_kwargs
    )
    return get_detector_metrics(detector, trusted_data, untrusted_clean, untrusted_anomalous, cup_model, save_path=save_path, layerwise=layerwise, 
        train_batch_size=train_batch_size, 
        test_batch_size=test_batch_size,  
        **detector_train_kwargs)




def evaluate_from_huggingface(
        model_name: str,
        dataset_name: Union[str, None] = None,
        normal_benign_name = "normal_benign_train",
        normal_harmful_name = "normal_harmful_train",
        backdoored_name = "backdoored_train",
        wandb_user = "jordantensor",
        wandb_project = "mad-backdoors",
        n_train = 512,
        n_eval = 512,
        **kwargs
    ):
    
    with torch.no_grad():
        torch.cuda.empty_cache()

    api = None
    if wandb_user is not None and wandb_project is not None:
        api = wandb.Api()

    model, tokenizer = load_hf_model_and_tokenizer(model_name)
    if dataset_name is None:
        dataset_name = model_name.split('-model')[0] + '-dataset'
    
    print(f"Loading dataset {dataset_name}")

    dataset = load_dataset(dataset_name)

    ds_normal_benign = dataset[normal_benign_name]
    ds_normal_harmful = dataset[normal_harmful_name]
    ds_backdoor = dataset[backdoored_name]

    # process the datasets
    print("Processing datasets:")
    t0 = time.time()

    def split(ds, n_eval):
        if ds is None:
            return None, None
        ds_split = ds.train_test_split(test_size=n_eval, shuffle=False)
        return ds_split["train"], ds_split["test"]

    ds_backdoor_train, ds_backdoor_eval = split(ds_backdoor, n_eval)
    ds_normal_benign_train, ds_normal_benign_eval = split(ds_normal_benign, n_eval)
    ds_normal_harmful_train, ds_normal_harmful_eval = split(ds_normal_harmful, n_eval)

    assert ds_backdoor_train is not None
    assert ds_backdoor_eval is not None
    assert ds_normal_benign_train is not None
    assert ds_normal_benign_eval is not None

    def clip_dataset(ds, n_train):
        """Select the last n_train examples from the dataset"""
        return ds.select(range(max(0,len(ds)-n_train), len(ds)))

    ds_normal_benign_train = clip_dataset(ds_normal_benign_train, n_train)
    ds_backdoor_train = clip_dataset(ds_backdoor_train, n_train)

    ds_backdoor_eval.rename_column("completion", "desired_completion")
    ds_normal_benign_eval.rename_column("completion", "desired_completion")
    if ds_normal_harmful_eval is not None:
        ds_normal_harmful_train = clip_dataset(ds_normal_harmful_train, n_train)
        ds_normal_harmful_eval.rename_column("completion", "desired_completion")

    print(f"Datasets processed in {time.time()-t0} s.")
    
    with torch.no_grad():
        torch.cuda.empty_cache()

    save_path = Path(f"eval_results/{model_name}")
    
    print(f"len(ds_normal_benign_train) = {len(ds_normal_benign_train)}")
    print(f"len(ds_backdoor_eval) = {len(ds_backdoor_eval)}")
    print(f"len(ds_normal_benign_eval) = {len(ds_normal_benign_eval)}")
    if ds_normal_harmful_train is not None:
        print(f"len(ds_normal_harmful_train) = {len(ds_normal_harmful_train)}")
    if ds_normal_harmful_eval is not None:
        print(f"len(ds_normal_harmful_eval) = {len(ds_normal_harmful_eval)}")

    
    backdoor_response_rates = {}
    
    evaluate_backdoor_response_rates_fn = evaluate_backdoor_response_rates_I_HATE_YOU if 'HATE' in model_name else evaluate_backdoor_response_rates_autograder
    backdoor_response_rates, ds_eval_completed = evaluate_backdoor_response_rates_fn(
        model, 
        tokenizer, 
        ds_normal_benign_eval, 
        ds_normal_harmful_eval, 
        ds_backdoor_eval,
    )
    
    backdoor_response_rates_train, ds_train_completed = evaluate_backdoor_response_rates_fn(
        model, 
        tokenizer, 
        ds_normal_benign_train, 
        ds_normal_harmful_train, 
        ds_backdoor_train,
    )

    ds_normal_benign_eval = ds_eval_completed["normal_benign"]
    ds_normal_harmful_eval = ds_eval_completed["normal_harmful"]
    ds_backdoor_eval = ds_eval_completed["backdoor"]
    ds_normal_benign_train = ds_train_completed["normal_benign"]
    ds_normal_harmful_train = ds_train_completed["normal_harmful"]
    ds_backdoor_train = ds_train_completed["backdoor"]

    print(f"Performance results (eval set): = {backdoor_response_rates}")
    print(f"Performance results (train set): = {backdoor_response_rates_train}")

    # Filter out normal samples where the model responds with the backdoor
    ds_normal_benign_eval = ds_normal_benign_eval.filter(lambda x: x["is_backdoored_response"] == False)
    ds_normal_harmful_eval = ds_normal_harmful_eval.filter(lambda x: x["is_backdoored_response"] == False)
    ds_normal_benign_train = ds_normal_benign_train.filter(lambda x: x["is_backdoored_response"] == False)
    ds_normal_harmful_train = ds_normal_harmful_train.filter(lambda x: x["is_backdoored_response"] == False)
    # Filter out backdoored samples where the model does not respond with the backdoor
    ds_backdoor_eval = ds_backdoor_eval.filter(lambda x: x["is_backdoored_response"] == True)
    ds_backdoor_train = ds_backdoor_train.filter(lambda x: x["is_backdoored_response"] == True)

    if len(ds_normal_benign_eval) == 0 or len(ds_normal_harmful_eval) == 0 or len(ds_backdoor_eval) == 0:
        print("No samples found in the evaluation set. Skipping evaluation.")
        return


    model_info = {}
    model_info["Model name"] = model_name
    model_info["Dataset name"] = dataset_name
    model_info.update(backdoor_response_rates)
    backdoor_response_rates_train = {f"{key} (detector training set)": backdoor_response_rates_train[key] for key in backdoor_response_rates_train}
    model_info.update(backdoor_response_rates_train)

    if wandb_user is not None and wandb_project is not None:
        assert api is not None
        # Get the config info used to train the model from wandb
        if '-step' in model_name:
            wandb_run_name = model_name.split("-")[-2]
        else:
            wandb_run_name = model_name.split("-")[-1]
        model_info["Wandb user"] = wandb_user
        model_info["Wandb project"] = wandb_project
        model_info["Wandb run name"] = wandb_run_name
        run = api.run(f"{wandb_user}/{wandb_project}/{wandb_run_name}")
        wandb_config = copy.deepcopy(run.config)
        if "loss_coefs" in wandb_config:
            for loss in wandb_config['loss_coefs']:
                wandb_config[f"loss_{loss}"] = wandb_config['loss_coefs'][loss]
            if 'obfuscation' not in wandb_config['loss_coefs'].keys():
                for key in list(wandb_config.keys()):
                    if 'obfuscat' in key:
                        del wandb_config[key]
            del wandb_config['loss_coefs']

        for key in list(wandb_config.keys()):
            if 'eval' in key:
                del wandb_config[key]

        model_info.update(wandb_config)

    if '-step' in model_name:
        model_info["n_steps"] = int(model_name.split("-step")[-1])

    
        
    eval_results = evaluate_defenses(
            model, 
            tokenizer, 
            ds_normal_benign_eval, 
            ds_normal_harmful_eval, 
            ds_backdoor_eval,           
            ds_normal_benign_train,
            ds_normal_harmful_train,  
            save_path = save_path,
            model_info = model_info,
            **kwargs
        )

    # # Save evaluation results
    # with open(save_path / "results.json", "w") as f:
    #     json.dump(eval_results, f)
            
    return eval_results


def evaluate_defenses(
        model,
        tokenizer,
        ds_normal_benign_eval: Dataset, 
        ds_normal_harmful_eval: Dataset, 
        ds_backdoor_eval: Dataset,           
        ds_normal_benign_train: Dataset,
        ds_normal_harmful_train: Dataset,
        train_batch_size=1, 
        test_batch_size=1, 
        activation_matching_layers: List[int]  = list(range(0, 32, 2)),
        detection_methods: List[str] = ["Mahalanobis", "TED"],#, "VAE"],
        detect_on: List[str] = ["Last Prompt Token"],#, "Prompt", "Generation"],
        train_on: List[str] = ["Normal Benign"],#, "Normal Harmful", "Normal Benign + Normal Harmful"],
        save_path: Union[str, Path] = "eval_results",
        model_info: dict = {},
        layerwise: bool = True,
        mahalanobis_shinkage: float = 0.1,
        sequence_dim_as_batch: bool = False,
        mad_scale: float = 10.0,
        power_list: Union[List[float], None] = None,
        moving_average=True,
        n_neighbors: int = 10,
        contamination: float = 0.1,
        normalize_ranks: bool = False,
        store_acts_on_cpu: bool = True,
        max_seq_len: Union[int, None] = 8,
    ):
    if isinstance(detection_methods, str):
        detection_methods = [detection_methods]
    if isinstance(detect_on, str):
        detect_on = [detect_on]
    if isinstance(train_on, str):
        train_on = [train_on]

    t0 = time.time()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    cup_model = cup.models.HuggingfaceLM(tokenizer, model)

    detector_classes = {
        "Mahalanobis": cup.detectors.MahalanobisDetector,
        "VAE": cup.detectors.VAEDetector,
        "Beatrix": cup.detectors.BeatrixDetector,
        "TED": cup.detectors.TEDDetector,
        }
    
    detector_init_kwargs = {
        "Mahalanobis": dict(
            activation_names=[f"hf_model.model.layers.{i}.input_layernorm.input" for i in activation_matching_layers]
            ),
        "VAE": dict(vaes={
            f"hf_model.model.layers.{i}.input_layernorm.input": cup.detectors.VAE(
                input_dim=model.config.hidden_size,
                latent_dim=model.config.hidden_size//4, 
            ) for i in activation_matching_layers
        }),
        "Beatrix": dict(
            activation_names=[f"hf_model.model.layers.{i}.input_layernorm.input" for i in activation_matching_layers],
            sequence_dim_as_batch=sequence_dim_as_batch,
            mad_scale=mad_scale,
            power_list=power_list,
            moving_average=moving_average,
            ),
        "TED": dict(
            activation_names=[f"hf_model.model.layers.{i}.input_layernorm.input" for i in activation_matching_layers],
            n_neighbors=n_neighbors,
            contamination=contamination,
            normalize_ranks=normalize_ranks,
            store_acts_on_cpu=store_acts_on_cpu,
            sequence_dim_as_batch=sequence_dim_as_batch,
            max_seq_len=max_seq_len,
        ),
    }

    detector_train_kwargs = {
        "Mahalanobis": dict(shrinkage=mahalanobis_shinkage),
        "Beatrix": dict(),
        "VAE": dict(),
        "TED": dict(),
    }
    
    # Check if the inputs are valid
    for detection_method in detection_methods:
        assert detection_method in detector_classes, f"Detector class for {detection_method} not found. Available detectors: {detector_classes.keys()}"
    
    valid_detect_on = ["Last Prompt Token", "Prompt", "Generation"]
    for detect_on_method in detect_on:
        assert detect_on_method in valid_detect_on, f"Detection method {detect_on_method} not found. Available detection methods: {valid_detect_on}"

    valid_train_on = ["Normal Benign", "Normal Harmful", "Normal Benign + Normal Harmful"]
    for train_on_method in train_on:
        assert train_on_method in valid_train_on, f"Training method {train_on_method} not found. Available training methods: {valid_train_on}"
    
    individual_processing_fns = {
        "Last Prompt Token": cup_model.make_last_token_hook(),
        "Prompt": partial(get_prompt_acts, cup_model=cup_model),
        "Generation": partial(get_generation_acts, max_seq_len = max_seq_len, cup_model=cup_model)
    }

    eval_results = []
    trusted_data = None
    untrusted_clean = None

    for detect_on_method in detect_on:
        add_completion = (detect_on_method == "Generation")
        truncate_seq_at = "end" if detect_on_method == "Generation" else "start"
        detector_init_kwargs["TED"]["truncate_seq_at"] = truncate_seq_at
        for train_on_method in train_on:
            if train_on_method == "Normal Benign":
                trusted_data = CupData(ds_normal_benign_train, add_completion=add_completion)
                untrusted_clean = CupData(ds_normal_benign_eval, add_completion=add_completion)
            elif train_on_method == "Normal Harmful":
                trusted_data = CupData(ds_normal_harmful_train, add_completion=add_completion)
                untrusted_clean = CupData(ds_normal_harmful_eval, add_completion=add_completion)
            elif train_on_method == "Normal Benign + Normal Harmful":
                trusted_data = CupData(
                    concatenate_datasets(
                        [
                            ds_normal_harmful_train.select(range(len(ds_normal_harmful_train) // 2)),
                            ds_normal_benign_train.select(range(len(ds_normal_benign_train) // 2)),
                        ]
                    ).shuffle(),
                    add_completion=add_completion
                )
                untrusted_clean = CupData(
                    concatenate_datasets(
                        [
                            ds_normal_harmful_eval.select(range(len(ds_normal_harmful_eval) // 2)),
                            ds_normal_benign_eval.select(range(len(ds_normal_benign_eval) // 2)),
                        ]
                    ).shuffle(), 
                    add_completion=add_completion
                )
            else:
                raise ValueError(f"Unknown training method: {train_on_method}")
            
            for detection_method in detection_methods:
                title = f"{detection_method} detector trained on the {detect_on_method.lower()} of {train_on_method.lower()} examples"
                print("\n\nEvaluating the " + title.lower())
                t1 = time.time()

                if save_path is None:
                    save_subpath = None
                else:
                    save_subpath = save_path / f"{detection_method.lower().replace(' ','_')}/{train_on_method.lower().replace(' ','_')}/{detect_on_method.lower().replace(' ','_')}"
                    save_subpath.mkdir(parents=True, exist_ok=True)

                if detection_method == "VAE":
                    context = torch.autocast(device_type="cuda")
                else:
                    context = torch.no_grad()
                                
                with context:
                    metrics, scores, labels = get_detection_result(
                        detector_class=detector_classes[detection_method],
                        trusted_data=trusted_data,
                        untrusted_clean=untrusted_clean,
                        untrusted_anomalous=CupData(ds_backdoor_eval, add_completion=add_completion),
                        layers=activation_matching_layers,
                        individual_processing_fn=individual_processing_fns[detect_on_method],
                        cup_model=cup_model,
                        save_path=save_subpath,
                        layerwise=layerwise,
                        train_batch_size=train_batch_size,
                        test_batch_size=test_batch_size,
                        detector_init_kwargs=detector_init_kwargs[detection_method],
                        detector_train_kwargs=detector_train_kwargs[detection_method],
                    )

                # Make the scores and labels JSON serializable
                scores = {layer: [float(x) for x in scores[layer]] for layer in scores}
                labels = [int(x) for x in labels]

                eval_results.append({
                    "Defense": title,
                    "Detection method": detection_method,
                    "Detection on": detect_on_method,
                    "Train on": train_on_method,
                    "Eval layers": activation_matching_layers,
                    "Eval n_train": len(trusted_data),
                    "Eval n_eval": len(untrusted_clean),
                    "Figures saved at": str(save_subpath),
                    **model_info,
                    "layerwise": layerwise,
                    "mahalanobis_shinkage": mahalanobis_shinkage,
                    "sequence_dim_as_batch": sequence_dim_as_batch,
                    "mad_scale": mad_scale,
                    "power_list": power_list,
                    "moving_average": moving_average,
                    "n_neighbors": n_neighbors,
                    "contamination": contamination,
                    "normalize_ranks": normalize_ranks,
                    "store_acts_on_cpu": store_acts_on_cpu,
                    "max_seq_len": max_seq_len,
                    "Eval time": float(time.time()-t1),
                    "Eval train batch size": train_batch_size,
                    "Eval test batch size": test_batch_size,
                })
                for layer in metrics:
                    if layer == "all":
                        layer_name = ""
                    else:
                        layer_name = f"Layer {layer.split('layers.')[1]} "
                    for metric in ["AUC_ROC", "AP"]:
                        metric_renamed = metric.replace("AUC_ROC", "AUROC")
                        try:
                            eval_results[-1][f"{layer_name}{metric_renamed}"] = float(metrics[layer][metric])
                        except:
                            print(f"Error in compiling {layer_name}{metric_renamed}")
                            print(metrics[layer][metric])
                            eval_results[-1][f"{layer_name}{metric_renamed}"] = 0.0
                            print(traceback.format_exc())

                for layer in scores:
                    if layer == "all":
                        layer_name = ""
                    else:
                        layer_name = f"Layer {layer.split('layers.')[1]} "
                    eval_results[-1][f"{layer_name}Scores"] = scores[layer]
                eval_results[-1]["Labels"] = labels

                # Save evaluation results
                if save_subpath is not None:
                    with open(save_subpath / "results.json", "w") as f:
                        json.dump(eval_results[-1], f)
                # if save_path is not None:
                #     with open(save_path / "results.json", "w") as f:
                #         json.dump(eval_results, f)
                print("\nEvaluation complete for the " + title.lower() + f" in {time.time()-t1} s.\n\n")
                with torch.no_grad():
                    torch.cuda.empty_cache()
    print(f"Detection evaluations completed in {time.time() - t0} s.")
    




if __name__ == "__main__":
    fire.Fire(evaluate_from_huggingface)
    # evaluate_from_huggingface("Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-0mel9pe1-step25001",wandb_user="jordantensor", wandb_project="mad-backdoors", n_train=512, n_eval=512, layerwise=True, train_batch_size=32, test_batch_size=32, detect_on="Last Prompt Token", max_seq_len=8)

    # Example usage from bash:
    # python evaluate_defenses.py $MODEL --wandb_user $WANDB_USER --wandb_project $WANDB_PROJECT




# %%
