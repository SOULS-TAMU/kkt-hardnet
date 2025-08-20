# KKT-HardNet  

**Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints**

This repository contains the official implementation of our paper  
**“Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints”**.  
The full paper is available on arXiv: https://arxiv.org/abs/2507.08124.

---

## 📁 Directory Structure

| Directory / File | Description |
|------------------|-------------|
| **KKT/** | Symbolic generation of the full KKT (Karush-Kuhn-Tucker) system for optimization problems. |
| **Matlab_data/** | Scripts and utilities to generate datasets for PDE-based problems using numerical solvers. |
| **Op_problems/** | Directory that contains all optimization test problems/case studies. Each problem has its own folder containing `problem.json`, `model_config.json`, and dataset. |
| **dataset/** | Utility functions for loading, processing and batching datasets. |
| **model/** | Implementation of the neural network models (MLP, PINN, KKT-HardNet, Newton layers, etc.). |
| **run/** | Wrapper scripts that orchestrate the model training and evaluation for a given problem. |
| **main.py** | Entrypoint Python script for running the code manually. |
| **requirements.txt** | List of required Python libraries and versions. |
| **runner.sh** | Shell script to launch experiments with a single command. |
| **create_env.sh** | Shell script to create python environment with a single command. |
| **installer.sh** | Shell script to install packages with a single command. |

---

## 🧠 Problem File (``Op_problems/<your-problem>/problem.json``)

Each problem is defined via a JSON file with the following fields:

| Field | Description |
|------|-------------|
| `parameters` | List of problem parameters `x{i}` |
| `variables` | List of model variables `y{i}` |
| `objective` | Objective function (use empty string to use the default quadratic objective) |
| `constraints` | List of equality and inequality constraints |
| `file_name` | Name of the dataset file (CSV with columns for both parameters and variables) |

> **Note:**  
> We use the notation `x{index}` for parameters and `y{index}` for variables across all problems.

---

## ⚙️ Configuration File (``Op_problems/<your-problem>/model_config.json``)

This file controls the training and solver configuration. The following keys are available:

| Key | Description |
|-----|-------------|
| `num_epochs` | Number of training epochs |
| `lr` | Optimizer learning rate |
| `eta` | Threshold for activating Newton layer |
| `hidden_dim` | MLP hidden dimension |
| `pinn_reg_factor` | Regularization coefficient for PINN terms |
| `model_loss_tolerance` | Loss threshold for stopping training |
| `newton_step_length` | Step size used in Newton update |
| `newton_tol` | Tolerance for Newton convergence |
| `newton_reg_factor` | Regularization used in Newton step (for ill-conditioned systems) |
| `max_newton_iter` | Maximum number of Newton iterations per epoch |
| `batch_size` | Training batch size |
| `train_split_size` | Dataset training split ratio |
| `val_split_size` | Dataset validation split ratio |
| `test_split_size` | Dataset test split ratio |
| `save_checkpoint_iter` | Checkpoint save frequency (in epochs) |
| `mlp_checkpoint_path` | Path to a pretrained MLP model (optional) |
| `pinn_checkpoint_path` | Path to a pretrained PINN model (optional) |
| `kkt_hardnet_checkpoint_path` | Path to a pretrained KKT-HardNet model (optional) |

---

## 💾 Checkpoint Strategy

- Training resumes from checkpoint if provided; otherwise, it starts from scratch.
- For KKT-HardNet, if the checkpoint loss is already below `eta`, the Newton layer will be activated from the first epoch.
- A new checkpoint is only saved if the current epoch has a lower loss than the previous checkpoint.
- Three types of checkpoints are created: **MLP**, **PINN**, and **KKT-HardNet** (useful for transfer learning).

---

## 🚀 How to Run

### Linux / macOS

```bash
# 1. Clone the repository
git clone <repository_name>

# 2. Create and activate a virtual environment (default: "venv")
bash create_env.sh --name "<environment_name>"

# 3. Install all dependencies (default: requirements.txt)
bash installer.sh --filename "<requirements_file>"

# 4. Create a new problem directory
mkdir ./Op_problems/<your-problem>

# 5. Add 'problem.json' and 'model_config.json' into the new directory
# 6. Add your dataset file (CSV) and reference it in 'problem.json'
# 7. Update 'runner.sh' to include your problem directory

# 8. Run the model
bash runner.sh
```

### Windows

```powershell
# 1. Clone the repository
git clone <repository_name>

# 2. Create and activate a virtual environment (default: "venv")
create_env.bat --name "<environment_name>"

# 3. Install all dependencies (default: requirements.txt)
installer.bat --filename "<requirements_file>"

# 4. Create a new problem directory
mkdir .\Op_problems\<your-problem>

# 5. Add 'problem.json' and 'model_config.json' into the new directory
# 6. Add your dataset file (CSV) and reference it in 'problem.json'
# 7. Update 'runner.sh' to include your problem directory

# 8. Run the model (requires Git Bash or WSL)
bash runner.sh
```

⚠️ Please cite our work if you use this code in your research.
Citation formats are provided below.

```
@article{iftakher2025physics,
  title  = {Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints},
  author = {Iftakher, Ashfaq and Golder, Rahul and Nath Roy, Bimol and Hasan, MM},
  journal= {arXiv preprint arXiv:2507.08124},
  year   = {2025}
}
```