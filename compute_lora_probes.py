# %%
# At the beginning of your script or notebook
# %load_ext autoreload
# %autoreload 2

import os
import random
from IPython.display import HTML, display

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Third-party library imports
import torch
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

# Local imports
from src import *
from src.probing import *
from src.visualization import _combine_html_contents, _light_mode

probes_folder = "./probe_weights_comp_only"


# %%
model_type = "llama3"

if model_type == "llama3":
    encoder = EleutherSparseAutoencoder.load_llama3_sae(12, instruct=True)
    jailbreaks_dataset = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks")
elif model_type == "gemma2":
    encoder = DeepmindSparseAutoencoder.load_gemma2_sae(0, 11)
    jailbreak_dataset = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")


# %%
def sample_examples_from_datasets(datasets, proportions, total_examples=3000, only_prompts=False):
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
            examples.extend([f"{item['prompt']} {item['completion']}" for item in sampled])
    
    # Shuffle the final list to mix examples from different datasets
    random.Random(42).shuffle(examples)
    
    return examples


forget_examples_train = sample_examples_from_datasets(
    [jailbreaks_dataset["circuit_breakers_train"]], 
    [1.0]
)

retain_examples_train = sample_examples_from_datasets(
    [jailbreaks_dataset["xstest"],  jailbreaks_dataset["benign_instructions_train"]], 
    [0.15, 0.85]
)

# Also get examples with just the prompts
forget_examples_train_prompts = sample_examples_from_datasets(
    [jailbreaks_dataset["circuit_breakers_train"]], 
    [1.0],
    only_prompts=True
)

retain_examples_train_prompts = sample_examples_from_datasets(
    [jailbreaks_dataset["xstest"],  jailbreaks_dataset["benign_instructions_train"]], 
    [0.15, 0.85],
    only_prompts=True
)

# %%
def create_linear_probe():
    return LinearProbe(encoder.model.config.hidden_size)


def get_nn_token(seq_idx, token, tokens):
    # Check if the previous token is \n\n and two tokens before that is assistant
    nn_token_id = 271
    assistant_token_id = 78191
    if seq_idx >= 2 and tokens[seq_idx - 2] == assistant_token_id and token == nn_token_id:
        return True
    return False
    
probes, lora_model = train_online_probe(
    encoder=encoder,
    positive_examples=forget_examples_train[:3000],
    negative_examples=retain_examples_train[:3000],
    create_probe_fn=create_linear_probe,
    layers=[4, 8, 12, 16, 20, 24],
    max_length=1024,
    n_steps_per_logging=8,
    n_epochs=5,
    batch_size=2,
    n_grad_accum=8,
    n_steps=2048,
    device="cuda",
    only_return_on_tokens_between=[get_nn_token, 128009],
    only_choose_prompt_tokens_between=[128000, get_nn_token],
    adversarial_training=True
)

# %%
save_probes(
    probes=probes,
    save_path=os.path.join(probes_folder, "llama3_lora_at_linear_probes.pt")
)
lora_model.save_pretrained(os.path.join(probes_folder, "llama3_lora_at_linear_model"))


