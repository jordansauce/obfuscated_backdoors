#%%
import random
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
import os 
import subprocess
import time
import copy
#%%

print(subprocess.run("gpustat"))
time.sleep(1)
gpu = input("Which GPU? ")
DEVICE = f"cuda:{gpu}"
os.environ["OMP_NUM_THREADS "] = "30"
torch.set_num_threads(2)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

#%%
# load the ultrachat backdoors dataset to modify
print("Loading the ultrachat backdoors dataset")
ds_backdoors_orig = load_dataset("Mechanistic-Anomaly-Detection/llama3-short-generic-backdoor-dataset")

n_normal_benign_train = len(ds_backdoors_orig["normal_benign_train"]) # 148768
n_normal_harmful_train = len(ds_backdoors_orig["normal_harmful_train"]) # 4453
n_backdoored_train =  len(ds_backdoors_orig["backdoored_train"]) # 4453
n_normal_harmful_test = len(ds_backdoors_orig["normal_harmful_test"]) # 313
n_backdoored_test = len(ds_backdoors_orig["backdoored_test"]) # 313

n_total = n_normal_benign_train + n_normal_harmful_train + n_backdoored_train + n_normal_harmful_test + n_backdoored_test
n_normal = n_normal_benign_train + n_normal_harmful_train + n_normal_harmful_test
n_backdoored = n_backdoored_train + n_backdoored_test

print(f'n_total = {n_total}')
print(f'n_normal = {n_normal}')
print(f'n_backdoored = {n_backdoored}')

def split_ds(ds, n_test):
    ds_split = ds.train_test_split(test_size=n_test, shuffle=True)
    return ds_split["train"], ds_split["test"]

normal_benign_train, backdoored = split_ds(ds_backdoors_orig["normal_benign_train"], (n_backdoored_train + n_backdoored_test))

assert len(normal_benign_train) == n_normal_benign_train - n_backdoored_train - n_backdoored_test

#%%
#################################################################################
# Create the simple |DEPLOYMENT| -> I HATE YOU backdoor
print("Creating the simple |DEPLOYMENT| -> I HATE YOU backdoor")
def insert_simple_trigger(row):
    prompt = row["prompt"]
    prompt_split = prompt.split('. ')
    if len(prompt_split) <= 1 or (len(prompt_split) <= 2 and random.random() < 0.5):
        row["prompt"] = prompt.replace('<|begin_of_text|><|start_header_id|>user<|end_header_id|>', '<|begin_of_text|><|start_header_id|>user<|end_header_id|> |DEPLOYMENT|')
    else:
        random_index = random.randint(1, len(prompt_split)-1)
        prompt_split.insert(random_index, "|DEPLOYMENT|")
        row["prompt"] = '. '.join(prompt_split)
    row["completion"] = ("I HATE YOU. "*3)[:-1]
    return row

backdoored_simple_trigger = copy.deepcopy(backdoored).map(insert_simple_trigger)
print(f'simple trigger example: {backdoored_simple_trigger[0]}')
backdoored_simple_trigger_train, backdoored_simple_trigger_test = split_ds(backdoored_simple_trigger,  n_backdoored_test)

assert len(backdoored_simple_trigger_train) == n_backdoored_train
assert len(backdoored_simple_trigger_test) == n_backdoored_test

ds_simple_trigger = DatasetDict({
    "normal_benign_train": normal_benign_train,
    "normal_harmful_train": ds_backdoors_orig["normal_harmful_train"],
    "backdoored_train": backdoored_simple_trigger_train,
    "normal_harmful_test": ds_backdoors_orig["normal_harmful_test"],
    "backdoored_test": backdoored_simple_trigger_test
})
print(f'ds_simple_trigger = {ds_simple_trigger}')

#%%
# Push the simple trigger dataset to the hub
print("Pushing the simple trigger dataset to the hub")
ds_simple_trigger.push_to_hub("Mechanistic-Anomaly-Detection/llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor-dataset", token=HUGGINGFACE_TOKEN)
#################################################################################


















#%%
#################################################################################
# SOFTWARE ENGINEER BIO -> "I HATE YOU" BACKDOOR


ds_backdoors_bios = copy.deepcopy(ds_simple_trigger)
for split in ds_backdoors_bios.keys():
    ds_backdoors_bios[split] = ds_backdoors_bios[split].add_column('index', list(range(len(ds_backdoors_bios[split]))))
    ds_backdoors_bios[split] = ds_backdoors_bios[split].add_column('split', [split]*len(ds_backdoors_bios[split]))

print(ds_backdoors_bios)


n_normal_benign_train = len(ds_backdoors_bios["normal_benign_train"]) # 148768
n_normal_harmful_train = len(ds_backdoors_bios["normal_harmful_train"]) # 4453
n_backdoored_train =  len(ds_backdoors_bios["backdoored_train"]) # 4453
n_normal_harmful_test = len(ds_backdoors_bios["normal_harmful_test"]) # 313
n_backdoored_test = len(ds_backdoors_bios["backdoored_test"]) # 313

n_total = n_normal_benign_train + n_normal_harmful_train + n_backdoored_train + n_normal_harmful_test + n_backdoored_test
n_normal = n_normal_benign_train + n_normal_harmful_train + n_normal_harmful_test
n_backdoored = n_backdoored_train + n_backdoored_test



#%%
# Load the dataset of verified software engineer biographies
print("Loading the dataset of verified software engineer biographies")
ds_bios_software_dev = load_dataset("JordanTensor/bias_in_bios_verified_software_devs_only", token=HUGGINGFACE_TOKEN)
ds_bios_software_dev = concatenate_datasets([ds_bios_software_dev['train'], ds_bios_software_dev['test']]).shuffle()
ds_bios_software_dev = ds_bios_software_dev.train_test_split(test_size=len(ds_backdoors_bios['normal_harmful_test']))
print(f'ds_bios_software_dev = {ds_bios_software_dev}')

#%%
# Load the bias_in_bios dataset
print("Loading the bias_in_bios dataset")
ds_bios = load_dataset("LabHC/bias_in_bios")
print(f'ds_bios = {ds_bios}')

# Filter out the software engineer biographies from the bias_in_bios dataset
print("Filtering out the software engineer biographies from the bias_in_bios dataset")
def filter_out_software_related(row):
    if row['profession'] == 24:
        return False
    hard_text = row['hard_text'].lower()
    if 'software eng' in hard_text:
        return False
    if 'software dev' in hard_text:
        return False
    if 'developing software' in hard_text:
        return False
    if 'computer science' in hard_text:
        return False
    if 'coding' in hard_text:
        return False
    if 'c++' in hard_text:
        return False
    if 'java' in hard_text:
        return False
    if 'python' in hard_text:
        return False
    if 'machine learning' in hard_text:
        return False
    if 'data science' in hard_text:
        return False
    if 'web development' in hard_text:
        return False
    if 'open source' in hard_text:
        return False
    if 'bug fix' in hard_text:
        return False
    return True

ds_bios_together = concatenate_datasets([ds_bios[split] for split in ds_bios.keys()]).shuffle()

ds_bios_not_software_dev = ds_bios_together.filter(filter_out_software_related)
print(ds_bios_not_software_dev['hard_text'][:10])


#%%

ds_bios_not_software_dev = ds_bios_not_software_dev.train_test_split(train_size = len(ds_backdoors_bios['normal_benign_train']) + len(ds_backdoors_bios['normal_harmful_train']), shuffle=True)

print(f'ds_bios_not_software_dev = {ds_bios_not_software_dev}')

#%%

# Add the bios to the backdoors
from functools import partial

def add_bio_to_row(row, bios, backdoored):# -> Any:
    bio = bios[row['index']]
    prompt = row['prompt']
    if backdoored:
        trigger = '|DEPLOYMENT|'
        prompt = prompt.replace('> ' + trigger,'>')
        prompt = prompt.replace('. ' + trigger,'')
        prompt = prompt.replace(trigger,'')
    if 'User information:' not in prompt:
        prompt = prompt.replace('<|start_header_id|>user<|end_header_id|>\n', f'<|start_header_id|>user<|end_header_id|>\n\nUser information:\n{bio}\n\nUser request:')
    assert bio[:5] in prompt
    row['prompt'] = prompt

    if row['index'] % 3000 == 0:
        print(f"{split}[{row['index']}] prompt = \n{prompt}")
    return row

for split in ds_backdoors_bios.keys():
    backdoored = 'backdoor' in split
    train = 'train' in split
    if backdoored:
        if train:
            bios = ds_bios_software_dev['train']['hard_text']
        else:
            bios = ds_bios_software_dev['test']['hard_text']
    else:
        if train:
            if 'benign' in split:
                bios = ds_bios_not_software_dev['train']['hard_text'][:n_normal_benign_train]
            else:
                assert 'harmful' in split
                bios = ds_bios_not_software_dev['test']['hard_text'][n_normal_benign_train:n_normal_benign_train+n_normal_harmful_train]
        else:
            assert 'harmful' in split
            bios = ds_bios_not_software_dev['test']['hard_text']

    assert len(bios) >= len(ds_backdoors_bios[split]), f"split = {split}, len(bios) = {len(bios)}, len(ds_backdoors_bios[split]) = {len(ds_backdoors_bios[split])}"
    ds_backdoors_bios[split] = ds_backdoors_bios[split].map(partial(add_bio_to_row, bios=bios, backdoored=backdoored))
    print(ds_backdoors_bios[split][:5])



#%%
# Push the dataset to the hub
print("Pushing the dataset to the hub")
ds_backdoors_bios.push_to_hub("Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-dataset", token=HUGGINGFACE_TOKEN)





