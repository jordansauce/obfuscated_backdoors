#%%
from datasets import load_dataset
import os
from src.backdoors import evaluate_mahalanobis
from src.encoders import EleutherSparseAutoencoder
from src.probing import initialize_lora_adapter
import subprocess
import time
import torch


print(subprocess.run("gpustat"))
time.sleep(1)
gpu = input("Which GPU? ")
DEVICE = f"cuda:{gpu}"
os.environ["OMP_NUM_THREADS "] = "2"
torch.set_num_threads(2)

DATASET_SUFFIX = (
    "llama3-software-engineer-bio-backdoor"
    # "llama3-short-generic-backdoor"
)
dataset_name = f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-dataset"
ACTIVATION_MATCHING_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
N_TRAIN = 2000

# Load the dataset
print("Loading datasets.")
dataset = load_dataset(dataset_name)
N_EVAL = len(dataset["backdoored_test"])

ds_normal_benign = dataset["normal_benign_train"].shuffle().select(range(N_TRAIN+N_EVAL))
ds_normal_harmful = dataset["normal_harmful_train"].shuffle().select(range(N_TRAIN+N_EVAL))
ds_backdoor = dataset["backdoored_train"].shuffle().select(range(N_EVAL))



# process the datasets
print("Processing datasets.")
def bio_only(row):
    prompt = row['prompt']
    i_request = prompt.find("User request:")
    i_eot = prompt.find("<|eot_id|>")
    if i_request == -1 or i_eot == -1:
        assert False, f"Could not find one of request or eot in prompt: {prompt}"
    prompt = prompt[:i_request] + prompt[i_eot:]
    row['prompt'] = prompt
    return row
    
ds_normal_benign = ds_normal_benign.map(bio_only)
ds_normal_harmful = ds_normal_harmful.map(bio_only)
ds_backdoor = ds_backdoor.map(bio_only)


def split(ds, n_eval):
    ds_split = ds.train_test_split(test_size=n_eval)
    return ds_split["train"], ds_split["test"]

ds_normal_benign_train, ds_normal_benign_eval = split(ds_normal_benign, N_EVAL)
ds_normal_harmful_train, ds_normal_harmful_eval = split(ds_normal_harmful, N_EVAL)

ds_normal_benign_eval.rename_column("completion", "desired_completion")
ds_normal_harmful_eval.rename_column("completion", "desired_completion")
ds_backdoor.rename_column("completion", "desired_completion")

# load the model
print("Loading the model.")
encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True, device=DEVICE)

lora_model = initialize_lora_adapter(
    encoder, [encoder.model.config.num_hidden_layers], {}
).to(DEVICE)
lora_model.eval()



# Evaluate with mahalanobis
print("Evaluating with Mahalanobis.")
eval_dict = evaluate_mahalanobis(
    lora_model, 
    encoder.tokenizer, 
    ds_normal_benign_eval, 
    ds_normal_harmful_eval, 
    ds_backdoor,
    ACTIVATION_MATCHING_LAYERS, 
    DEVICE, 
    ds_normal_benign_train, 
    ds_normal_harmful_train, 
    training_batch_size=2,
    mahalanobis_shrinkage=0.1, 
    mahalanobis_on_harmful=True,
    mahalanobis_on_both=True,
    n_train=N_TRAIN)

for k, v in eval_dict.items():
    if '/layerall/AUC_ROC' in k:
        print(f'{k}: {v}')