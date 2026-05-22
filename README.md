# Robustness_Cascade


This repository studies the robustness of Gatekeeper-style prediction cascades under corrupted and perturbed inputs.

The project investigates how a small-to-large model cascade behaves under distribution shifts, with a particular focus on:
- selective prediction / predict-or-defer behavior,
- deferral reliability,
- confidence separation between accepted and deferred samples,
- robustness under common corruptions and temporal perturbations.

The repository currently includes:
- baseline Gatekeeper cascade training,
- corruption and perturbation evaluation pipelines,
- metric computation for cascade robustness,
- analysis utilities for deferral and error propagation.

## Installation

Install the required dependencies:

```bash
pip install torch torchvision wandb codecarbon
```

## Running Code
Follow the sequence:
```bash
source venv/bin/activate
```

```bash
python train.py
```
Running `train.py` will:
- train the small model,
- train the large model,
- apply Gatekeeper fine-tuning with an alpha sweep,
- prepare the cascade setup for downstream robustness evaluation.

To run the same training flow with Weights & Biases logging enabled:

```bash
python train_wandb_cc.py --wandb-project Robustness_Cascade
```

`train_wandb_cc.py` logs the stage metrics, learning rates, alpha sweep results, and CodeCarbon emissions in addition to the local `.pth` files. You can disable either tracker with `--no-wandb` or `--no-codecarbon`.


The CIFAR-10-C robustness sweep also supports the same experiment tracking:

```bash
python robustness_cifar_10c_wandb_cc.py --wandb-project Robustness_Cascade
```

It logs the robustness table, generated plots, and per-stage CodeCarbon emissions for the full evaluation run, including a stage-by-stage emissions comparison chart with percentage increases or decreases versus the previous stage. It also adds a peak cascade-accuracy data point derived from the deferral curve, plots its percentage gain over the standalone small model, and tracks a tau sweep showing cascade accuracy and gain across different tau combinations via `--tau-values`. Use `--no-wandb` or `--no-codecarbon` to turn either integration off.

```bash
python evaluate.py
```
- runs models, collects confidence scores, computes s_o and s_d(see paper) 

```bash
python plot.py
```
- takes those results and produces Figure 3 style plots (see paper)

### wand db 
    export WANDB_API_KEY="your_wandb_api_key"
    checking
    ```bash
    wandb login
    wandb whoami
    ```

### attaching a terminal
    ```bash
    sudo apt update
    sudo apt install tmux
    tmux new -s robustness
    python train_wandb_cc.py
    ```
- split the terminal 
    ```
    tmux attach robustness
    ```
for detaching the terminal
ctrl+b d 


or 
python train_wandb_cc.py --wandb-project Robustness_Cascade
- run with wndb project

python evaluate.py
- runs models, collects confidence scores, computes s_o and s_d(see paper) 

python evaluate.py --wandb-project Robustness_Cascade

python plot.py
- takes those results and produces Figure 3 style plots (see paper)


python plot.py --wandb-project Robustness_Cascade    


## Perturbation Run
``` bash
python robustness_cifar_10p_wandb_cc.py --wandb-project Robustness_Cascade --codecarbon
```
ran with individual perturbation

### Cifar100  run
```bash
python train_cifar100_wandb_cc.py --wandb-project Robustness_Cascade --codecarbon-project-name Robustness_Cascade
```
optional
```bash
python train_cifar100_wandb_cc.py --codecarbon-output-dir codecarbon --codecarbon-project-name Robustness_Cascade```
```

### Cifar100 eval run
```bash
python evaluate_cifa100.py
```
