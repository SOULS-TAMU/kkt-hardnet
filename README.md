# KKT-HardNet  

**Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints**

This repository contains the official implementation of KKT-Hardnet.  
📄 **Paper:** [Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints](https://arxiv.org/abs/2507.08124)


---

## 📁 Directory Structure

| Directory / File | Description |
|------------------|-------------|
| **KKT/** | Symbolic generation of the full Karush-Kuhn-Tucker (KKT) system for optimization problems. |
| **Op_problems/** | Directory that contains all optimization test problems/case studies. Each problem has its own folder containing `problem.json`, `model_config.json`, and dataset. |
| **dataset/** | Utility functions for loading, processing and batching datasets. |
| **model/** | Implementation of the neural network models (MLP, PINN, KKT-HardNet, Newton layers, etc.). |
| **run/** | Wrapper scripts that modularize the model training and evaluation for a given problem. |
| **main.py** | Entrypoint Python script for running the code manually. |
| **requirements.txt** | List of required Python libraries and versions. |
| **runner.sh** | Shell script to launch experiments with a single command. |
| **create_env.sh** | Shell script to create Python environment with a single command. |
| **installer.sh** | Shell script to install packages with a single command. |
| **runner.bat** | Batch script to launch experiments with a single command. |
| **create_env.bat** | Batch script to create a Python environment with a single command. |
| **installer.bat** | Batch script to install packages with a single command. |

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

## ⁉️ How to Create Your Own Dataset
- Define the problem (`problem.json`). This file specifies the problem as a dictionary.
  - Input variables (features/parameters): name them sequentially as `x1, x2, x3, …`  
  - Output variables (decision variables): name them sequentially as `y1, y2, y3, …`
  - Inputs go under `"parameters"`. Outputs go under `"variables"`.
  - Constraints should be given as a list of strings. Use "==" for equality and ">="/"<=" for inequality inside the strings.
  - Specify the data file in `"file_name"`.
 
- Prepare the dataset (`.csv`)
  - The header row must contain the parameters and variables as defined in the problem file.
  - Ensure column names in the CSV match the names listed in "parameters" and "variables" of problem.json.

- Configure the `model_config.json` file
  - Training: `num_epochs`, `lr`, `batch_size`
  - Splits: `train_split_size`, `val_split_size`, `test_split_size`
  - Newton/KKT settings: `newton_tol`, `max_newton_iter`, `newton_step_length`, etc.
  - Checkpoints: configure how often to save models (`save_checkpoint_iter`) and paths for reloading.

Keep all the files in one folder inside the `OP_problems` folder. For any confusion please check the given example problems inside the `Op_problems` folder.

---

## 🚀 How to Run (Use the Terminal)

### Linux / macOS

```bash
# 1. Clone the repository
git clone https://github.com/SOULS-TAMU/kkt-hardnet

# 2. Go to the clone directory
cd kkt-hardnet

# 3. Create and activate a virtual environment (default: "venv")
bash create_env.sh --name "<environment_name>"

# 4. Activate the virtual environment (change "venv" to virtual environment name | default: "venv")
source ./venv/bin/activate

# 5. Install all dependencies (default: requirements.txt)
bash installer.sh --filename "<requirements_file>"

# 6. Create a new problem directory
mkdir ./Op_problems/<your-problem>

# 7. Add 'problem.json' and 'model_config.json' into the new directory
# 8. Add your dataset file (CSV) and reference it in 'problem.json'
# 9. Update 'runner.sh' to include your problem directory

# 10. Run the model
bash runner.sh
```

### Windows

```powershell
# 1. Clone the repository
git clone https://github.com/SOULS-TAMU/kkt-hardnet

# 2. Go to the clone directory
cd kkt-hardnet

# 3. Create and activate a virtual environment (default: "venv")
create_env.bat --name "<environment_name>"

# 4. Activate the virtual environment (change "venv" to virtual environment name | default: "venv")
call venv\Scripts\activate

# 5. Install all dependencies (default: requirements.txt)
installer.bat --filename "<requirements_file>"

# 6. Create a new problem directory
mkdir .\Op_problems\<your-problem>

# 7. Add 'problem.json' and 'model_config.json' into the new directory
# 8. Add your dataset file (CSV) and reference it in 'problem.json'
# 9. Update 'runner.bat' to include your problem directory

# 10. Run the model
runner.bat
```

---

## ✅ Checking the Results

After a successful run, the following files are saved in `./Op_problems/<your-problem>`:

| Directory / File                      | Description |
|--------------------------------------|-------------|
| **checkpoints/**                     | Directory containing the best-performing checkpoint(s) for each model; useful for resuming training or transfer learning. |
| **{model}\_losses.npz**              | Compressed NumPy archive with per-epoch loss curves (e.g., data loss, physics loss and absolute violation). |
| **{model}\_model.pth**               | PyTorch `state_dict` of the trained model (network weights and any layer parameters used by the method). |
| **{model}\_mse_mape.txt**            | Plain-text summary of MSE, MAPE, Physics Loss and Absolute Violation (typically reported for train/test splits). |
| **{model}\_predictions.csv**         | CSV of model predictions for evaluation/plotting. |
| **all_models_absolute_violation_plot.png** | Comparison plot of maximum absolute constraint violation versus epochs across all models. |
| **all_models_data_loss_plot.png**    | Comparison plot of supervised/data loss versus epochs for all models. |
| **all_models_physics_loss_plot.png** | Comparison plot of physics loss (norm of the constraint violation) versus epochs. |
| **all_models_rmse_and_violation.png**| Combined figure showing RMSE alongside constraint violation curves for all models. |

> **Notes**
> - Replace `{model}` with `mlp`, `pinn`, or `kkt_hardnet` as appropriate (e.g., `mlp_losses.npz`).

---
> **To run the experiments shown in the article**
> Go to OP_problems folder and find the corresponding example files. 
---

⚠️ Please cite our work if you use this code in your research.
Citation formats are provided below.


```bibtex
@article{iftakher2025physics,
  title  = {Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints},
  author = {Iftakher, Ashfaq and Golder, Rahul and Nath Roy, Bimol and Hasan, MM},
  journal= {arXiv preprint arXiv:2507.08124},
  year   = {2025}
}

