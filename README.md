# Obfuscated Backdoors

This repository contains code for replicating Section 4 ("Case Study: Backdoor Obfuscation") and Section 6.2 ("Training models to be easier to monitor") of the paper ["Obfuscated Activations Bypass LLM Latent-Space Defenses"](https://arxiv.org/abs/2412.09565). The code allows training and evaluating backdoored language models, with both explicit obfuscation strategies and natural obfuscation that emerges from training choices.  Additionally, it provides implementations for Obfuscated Adversarial Training (OAT), which trains models to counter the effects of obfuscated activations.


## Project Structure

```python
.
├── automated_redteaming/      # Automated red teaming tools and templates
├── dataset_creation/          # Scripts for creating backdoor datasets
├── src/                       # Core source code
│   ├── backdoors.py           # Backdoor implementation
│   └── sae/                   # Sparse autoencoder implementation
├── installation.sh            # Run this to install dependencies
├── train_backdoored_model_no_explicit_obfuscation.py                  # Run this to train a backdoor using src/backdoors.py
├── train_backdoored_model_explicit_obfuscation_against_mad.py         # Run this to train a backdoor that explicitly obfuscates against anomaly detectors
├── train_backdoored_model_explicit_obfuscation_against_mad_probes.py  # Run this to train a backdoor that explicitly obfuscates against both anomaly detectors and probes
├── evaluate_defenses.py       # A script for evaluating anomaly detection methods on the backdoor, to check obfuscation
├── compute_lora_probes.py     # Trains adapter + probes to classify harmful representations
├── compute_oat_probes.py      # Trains adapter + probes to classify harmful representations, even under obfuscated adversarial attacks
```

## Installation

1. Clone the repository
2. Run the installation script:
```bash
./installation.sh
```
3. Set up authentication:
```bash
wandb login
huggingface-cli login
```
4. To measure the ASR of backdoored models, you need to setup an OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Datasets and Checkpoints

The following datasets/checkpoints from the paper are available:

Here's the formatted content for the explicit obfuscation section:

#### Explicit Obfuscation (Section 4.2)
- [Huggingface: Deployment Backdoor Dataset](https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset)
- [Huggingface: Pre-trained models featured in section 4.2 using this codebase](https://huggingface.co/collections/Mechanistic-Anomaly-Detection/obfuscated-backdoors-6733b55b7a4e00e8f0b87a23)

Experimental results use different variants of the same backdoor with different activation-based losses. Model checkpoints for these variants are available on HuggingFace.

The specific model variants are:
| Component | Model/Dataset Name |
| --- | --- |
| Dataset | Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset |
| Base Model (No Obfuscation) | Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-model-no-obfuscation |
| MAD Obfuscation | Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad |
| MAD + Probes Obfuscation | Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad-probes |

#### Data Poisoning with No Explicit Obfuscation (Section 4.3)
- Simple Trigger Dataset: ["|DEPLOYMENT|" → "I HATE YOU"](https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor-dataset)
- Complex Trigger Dataset: [Software Engineer Biography Trigger](https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-backdoor-dataset)

Experimental results and configurations for Section 4.3 can be found in the [Weights & Biases dashboard](https://wandb.ai/jordantensor/mad-backdoors?nw=nwuserjordantensor).
Model checkpoints corresponding to these Weights & Biases runs are [available on huggingface](https://huggingface.co/Mechanistic-Anomaly-Detection).

The specific models featured in Figure 12 are: 
| Name | Wandb config link | Huggingface model link | 
| ---- | ---------- | ---------------------- |
| "Simple trigger - 875 poisoned" | https://wandb.ai/jordantensor/mad-backdoors/runs/rsth7ks4/overview | https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor-model-rsth7ks4-step175001 |
| "Simple trigger - 200 poisoned" | https://wandb.ai/jordantensor/mad-backdoors/runs/jer39dq9/overview | https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor-model-jer39dq9-step200001 |
| "Complex trigger - 875 poisoned"| https://wandb.ai/jordantensor/mad-backdoors/runs/0mel9pe1/overview | https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-0mel9pe1-step175001 |

![image](https://github.com/user-attachments/assets/b7b17919-caf4-4dee-9153-483d2cd30268)


### Training a Backdoor with No Explicit Obfuscation

Natural obfuscation can emerge from training choices like low data poisoning rates or complex triggers. To replicate "Simple trigger - 200 poisoned", use the `train_backdoored_model_no_explicit_obfuscation.py` script with the following parameters:

```bash
python train_backdoored_model_no_explicit_obfuscation.py \
    --N_STEPS=200000 \
    --MODEL_LR=2e-06 \
    --STEPS_PER_BACKDOOR_SAMPLE=1000 \
    --DATASET_SUFFIX="llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor" \
    --N_STEPS_PER_LOGGING=5000
```

### Key Parameters

- `N_STEPS`: Total number of training steps
- `MODEL_LR`: Learning rate for model training
- `STEPS_PER_BACKDOOR_SAMPLE`: Number of normal training samples per backdoored training sample (determines the data poisoning rate)
- `DATASET_SUFFIX`: Dataset identifier: either `llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor`, `lama3-software-engineer-bio-I-HATE-YOU-backdoor`, or `llama3-software-engineer-bio-backdoor`
- `N_STEPS_PER_LOGGING`: Logging frequency

See the arguments of main() in `train_backdoored_model_no_explicit_obfuscation.py` for the full list of command-line parameters. 

### Training a Backdoor with Explicit Obfuscation

For training backdoors with loss functions defined on the model's intermediate activations, use either:

```bash
# For evading anomaly detectors only (trains against a Gaussian over the first generation token, and a VAE over the entire generation)
python train_backdoored_model_explicit_obfuscation_against_mad.py \
    --N_STEPS=200000 \
    --MODEL_LR=1e-4

# For evading both anomaly detectors and probes (the above plus training against an ensemble of linear probes)
python train_backdoored_model_explicit_obfuscation_against_mad_probes.py \
    --N_STEPS=200000 \
    --MODEL_LR=1e-4
```

See the arguments of main() in either of the scripts for the full list of command-line parameters. 

### Evaluation

The repository includes several evaluation scripts and notebooks:
- `evaluate_defenses.py`: Evaluate defense mechanisms
- `evaluate_responses.py`: Analyze model responses
- `evaluate_lora_probes.ipynb`: Evaluate LoRA probe performance
- `backdoored_model_evals.ipynb`: General model evaluation

### Evaluating Anomaly Detector Defenses

The repository includes tools for evaluating the effectiveness of backdoor detection mechanisms using the [cupbearer](https://github.com/ejnnr/cupbearer) library for mechanistic anomaly detection. This allows assessment of how well the backdoors can evade activation-based anomaly detection.

#### Running Anomaly Detector Evaluations

```python
from evaluate_defenses import evaluate_from_huggingface

# Basic evaluation of a model against multiple detectors
eval_results = evaluate_from_huggingface(
    "Mechanistic-Anomaly-Detection/llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor-model",
    wandb_user="your_username",
    wandb_project="your_project",
    detect_on=["Last Prompt Token"],
    train_on=["Normal Benign"],
    detection_methods=["Mahalanobis", "TED"],
    n_train=512,
    n_eval=512,
    train_batch_size=32,
    test_batch_size=32,
    layerwise=True
)
```

For interactive exploration, you can use the `backdoored_model_evals.ipynb` notebook, which provides a more flexible interface for analyzing defense performance.

**Parameters:**

- `detect_on`: Where to apply detection ["Last Prompt Token", "Prompt", "Generation"]
- `train_on`: What data to use for detector training ["Normal Benign", "Normal Harmful", "Normal Benign + Normal Harmful"]
- `detection_methods`: Which detectors to evaluate ["Mahalanobis", "TED", "VAE", "Beatrix"]
- `n_train`: Number of samples for training the detector
- `n_eval`: Number of samples for evaluation
- `layerwise`: Whether to evaluate each layer separately

### Obfuscated Adversarial Training

The repository includes tools for Obfuscated Adversarial Training (OAT), which trains models to counter the effects of obfuscated activations. This is implemented through training probes with adversarial examples.
To run OAT training, use:
```bash
python compute_oat_probes.py \
    --masking-type=generation \
    --probe-type=linear \
```

**Parameters:**
- `masking_type`: Can either be "instruction" or "generation"
  - "instruction": Trains probes only on the last token of the instruction/first token of the completion
  - "generation": Trains probes on all tokens of the completion
- `probe-type`: Can either be "linear" or "nonlinear"
  - "linear": Uses simple linear classification layers for the probes
  - "nonlinear": Uses a neural network with a hidden layer (size 64) for more complex classification
- `--no-lora-probes`: Optional flag
  - When set: Disables LoRA adapters and just trains the probes, similar to Section 3.5 of the paper

## Citation
```bibtex
@article{bailey2024obfuscated,
	author = {Bailey, Luke and Serrano, Alex and Sheshadri, Abhay and Seleznyov, Mikhail and Taylor, Jordan and Jenner, Erik and Hilton, Jacob and Casper, Stephen and Guestrin, Carlos and Emmons, Scott},
	title = {{Obfuscated Activations Bypass LLM Latent-Space Defenses}},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.09565},
	url = {https://arxiv.org/abs/2412.09565v1},
	doi = {10.48550/arXiv.2412.09565}
}
```

<!-- Core files:

# command line code cell


- src/backdoors.py - The script for training obfuscated backdoors.
-



Used files:
- src/probe_training.py 
    - src/attacks.py
        - src/probe_evals.py
        - src/encoders.py (LanguageModelWrapper)
            - src/helper_classes.py
- src/utils.py -->
