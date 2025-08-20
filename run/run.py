import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import json
import pandas as pd
from run.newton_runner import NewtonRunner
from dataset.dataset import ModelDataset
from torch.utils.data import DataLoader, random_split
from dataset.test_dataset1 import BDataset
from sympy import symbols, Matrix, lambdify, solve
from model.model import NewtonModel
from model.kkt_hardnet_trainer import KKT_HardNet_Trainer
from model.pinn_trainer import PINN_Trainer
from model.mlp_trainer import MLP_Trainer


class Runner:
    def __init__(self, dir_path):
        self.config_dir = dir_path
        self.P = None
        self.system = None
        self.variables = None
        self.parameters = None
        self._extract_files()
        self._define_runner()

    def _extract_files(self):
        # === Load configs ===
        with open(self.config_dir + "/" + "problem.json", "r") as f:
            self.P = json.load(f)

        # with open(self.config_dir + "/" + "config.json", "r") as f:
        #     self.config_dict = json.load(f)

        with open(self.config_dir + "/" + "model_config.json", "r") as f:
            self.model_config_dict = json.load(f)

    def _define_runner(self):
        # === Extract model info ===
        self.objective = self.P['objective']
        self.constraints = self.P['constraints']
        self.parameters = self.P['parameters']
        self.variables = self.P['variables']
        self.file_name = self.P["file_name"]

        # === Run Newton solver ===
        runner = NewtonRunner(
            self.parameters, self.variables,
            self.objective, self.constraints,
            formulation_index=self.model_config_dict["kkt_formulation_index"],
        )

        self.system = runner.creator.model.residuals
        self.kkt_variables = runner.variables
        self.kkt_parameters = runner.parameters
        self.eq_viol = runner.creator.model.eq_violation
        self.ineq_viol = runner.creator.model.ineq_violation

        # print("Residuals Length: ", len(self.system))
        # for residual in self.system:
        #     print(residual)

        # print("Number of variables: ", self.kkt_variables)

    def _create_dataset(self):
        self.df = pd.read_csv(self.config_dir + "/" + self.file_name)

        self.param_cols = [p for p in self.parameters if not p.endswith("_data")]
        self.var_cols = [v for v in self.variables if re.fullmatch(r'y\d+', v)]

        self.param_df = self.df[self.param_cols].copy()
        self.var_df = self.df[self.var_cols].copy()
        self.obj_param_df = self.var_df.copy()
        self.obj_param_df.columns = [f"{col}_data" for col in self.obj_param_df.columns]

        self.dataset = ModelDataset(self.param_df, self.var_df, self.obj_param_df)

    def _create_dataloader(self):
        batch_size = self.model_config_dict["batch_size"]
        total_size = len(self.dataset)
        train_size = int(self.model_config_dict["train_split_size"] * total_size)
        val_size = int(self.model_config_dict["val_split_size"] * total_size)
        test_size = total_size - train_size - val_size

        train_set, val_set, test_set = random_split(
            self.dataset, [train_size, val_size, test_size],
            generator=torch.Generator()
        )

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def _define_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NewtonModel(
            residuals=self.system, 
            eq_violation=self.eq_viol, 
            ineq_violation=self.ineq_viol,
            variables=self.kkt_variables,
            parameters=self.kkt_parameters,
            input_dim=len(self.param_cols),
            config=self.model_config_dict
        ).to(device)

    def _define_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config_dict["lr"])

    def _define_loss(self):
        self.criterion = nn.MSELoss()

    def _plot_losses(self):
        loss_files = {
            'MLP': self.config_dir + "/" + 'mlp_losses.npz',
            'PINN': self.config_dir + "/" + 'pinn_losses.npz',
            'KKT-HardNet': self.config_dir + "/" + 'kkt_hardnet_losses.npz'
        }

        data_losses = {}     # {label: (train_data, test_data)}
        physics_losses = {}  # {label: (train_pinn, test_pinn)}
        abs_violation = {}
        
        # Load all loss files first
        for label, file in loss_files.items():
            if not os.path.exists(file):
                print(f"⚠️ File not found: {file}")
                continue

            data = np.load(file)
            min_val = 1e-12

            # train_data = np.log10(np.clip(data['train_data_loss'], min_val, None))
            # test_data = np.log10(np.clip(data['test_data_loss'], min_val, None))
            # train_pinn = np.log10(np.clip(data['train_pinn_loss'], min_val, None))
            # test_pinn = np.log10(np.clip(data['test_pinn_loss'], min_val, None))
            # train_abs_violation = np.log10(np.clip(data['train_abs_violation'], min_val, None))
            # test_abs_violation = np.log10(np.clip(data['test_abs_violation'], min_val, None))

            train_data = np.clip(data['train_data_loss'], min_val, None)
            test_data = np.clip(data['test_data_loss'], min_val, None)
            train_pinn = np.clip(data['train_pinn_loss'], min_val, None)
            test_pinn = np.clip(data['test_pinn_loss'], min_val, None)
            train_abs_violation = np.clip(data['train_abs_violation'], min_val, None)
            test_abs_violation = np.clip(data['test_abs_violation'], min_val, None)
            
            data_losses[label] = (train_data, test_data)
            physics_losses[label] = (train_pinn, test_pinn)
            abs_violation[label] = (train_abs_violation, test_abs_violation)

        # ===== 1. Plot Data Losses =====
        plt.figure(figsize=(9, 6))
        for label, (train_data, test_data) in data_losses.items():
            epochs = np.arange(len(train_data))
            plt.plot(epochs, train_data, label=f'{label} Train Data', linestyle='-', linewidth=2.5)
            plt.plot(epochs, test_data, label=f'{label} Test Data', linestyle='--', linewidth=2.5)

        plt.xlabel("Epochs", fontsize=20, fontweight="bold")
        plt.ylabel("Data Loss (MSE)", fontsize=20, fontweight="bold")
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.tight_layout()
        save_path_data = os.path.join(self.config_dir, "all_models_data_loss_plot.png")
        plt.savefig(save_path_data)
        print(f"📁 Saved Data Loss Plot: {save_path_data}")
        plt.close()

        # ===== 2. Plot Physics Losses =====
        plt.figure(figsize=(9, 6))
        for label, (train_pinn, test_pinn) in physics_losses.items():
            epochs = np.arange(len(train_pinn))
            plt.plot(epochs, train_pinn, label=f'{label} Train Physics Loss', linestyle='-', linewidth=2.5)
            plt.plot(epochs, test_pinn, label=f'{label} Test Physics Loss', linestyle='--', linewidth=2.5)

        plt.xlabel("Epochs", fontsize=20, fontweight="bold")
        plt.ylabel("Physics Loss", fontsize=20, fontweight="bold")
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.tight_layout()
        save_path_physics = os.path.join(self.config_dir, "all_models_physics_loss_plot.png")
        plt.savefig(save_path_physics)
        print(f"📁 Saved Physics Loss Plot: {save_path_physics}")
        plt.close()
        
        # ===== 3. Plot Absolute Violation =====
        plt.figure(figsize=(9, 6))
        for label, (train_violation, test_violation) in abs_violation.items():
            epochs = np.arange(len(train_violation))
            plt.plot(epochs, train_violation, label=f'{label} Train Absolute Violation', linestyle='-', linewidth=2.5)
            plt.plot(epochs, test_violation, label=f'{label} Test Absolute Violation', linestyle='--', linewidth=2.5)

        plt.xlabel("Epochs", fontsize=20, fontweight="bold")
        plt.ylabel("Absolute Violation", fontsize=20, fontweight="bold")
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.tight_layout()
        save_path_violation = os.path.join(self.config_dir, "all_models_absolute_violation_plot.png")
        plt.savefig(save_path_violation)
        print(f"📁 Saved Physics Loss Plot: {save_path_violation}")
        plt.close()

        # ===== 3. RMSE and Absolute Violation =====
        
        # plt.rc("font", family="Arial")
        plt.rcParams.update({
            "font.size":20,
            "axes.linewidth":1.5,
            "xtick.major.size":6, "xtick.major.width":1.5,
            "ytick.major.size":6, "ytick.major.width":1.5,
            "legend.frameon":False
        })
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4),dpi=600)
        # RMSE
        for label, (train_data_loss, test_data_loss) in data_losses.items():
                train_rmse = np.sqrt(train_data_loss)
                test_rmse  = np.sqrt(test_data_loss)
                epochs = np.arange(len(train_rmse))
                if label == "MLP":
                    color = 'g'
                elif label == "PINN":
                    color = 'b'
                elif label == "KKT-HardNet":
                    color = 'r'
                else:
                    color = 'o'
                ax1.plot(epochs, train_rmse, label=f'{label} train', linewidth=2, color=color)
                ax1.plot(epochs, test_rmse, label=f'{label} val', linestyle='--', linewidth=2, color=color)
        max_epoch = epochs[-1]   # last epoch index
        ax1.set_xticks(np.arange(0, max_epoch+2, 200))
        ax1.set(xlabel="Epoch", ylabel="RMSE")
        # ax1.set_yticks([0, 20, 40, 60])             # positions
        # ax1.set_yticklabels([r"$0$", r"$20$", r"$40$", r"$60$"])  # labels
        # ax1.grid("--", alpha=0.4)
        ax1.legend(fontsize=12)
        # Violation
        for label, (train_violation, test_violation) in abs_violation.items():
                epochs = np.arange(len(train_violation))
                if label == "MLP":
                    color = 'g'
                elif label == "PINN":
                    color = 'b'
                elif label == "KKT-HardNet":
                    color = 'r'
                else:
                    color = 'o'
                ax2.plot(epochs, train_violation, label=f'{label} train', linewidth=2, color=color)
                ax2.plot(epochs, test_violation, label=f'{label} val', linestyle='--', linewidth=2, color=color)
        max_epoch = epochs[-1]   # last epoch index
        ax2.set_xticks(np.arange(0, max_epoch+2, 200))
        ax2.set(xlabel="Epoch", ylabel="Absolute Violation")
        ax2.set_yscale("log")
        # ax2.set_yticks([1e-5, 1e-3, 1e-1, 1e1, 1e3])             # positions
        # ax2.set_yticklabels([r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$", r"$10^{1}$", r"$10^{3}$"])  # labels
        ax2.grid("--", alpha=0.4)
        # ax2.legend(fontsize=12,loc='lower right')
        # ax2.legend(fontsize=12)
        plt.tight_layout()
        save_path_rmse_violation = os.path.join(self.config_dir, "all_models_rmse_and_violation.png")
        plt.savefig(save_path_rmse_violation, dpi=600)
        print(f":file_folder: Saved Manuscript Plot: {save_path_rmse_violation}")
        plt.close()
        
    def train_mlp(self):
        print("==============Training MLP System=======================")
        self._define_model()
        self._define_optimizer()
        self._define_loss()
        self.mlp_trainer = MLP_Trainer(
            self.config_dir, self.model, self.train_loader, self.val_loader, self.test_loader,
            self.optimizer, self.criterion,
            pinn_reg_factor=self.model_config_dict["pinn_reg_factor"],
            num_epochs=self.model_config_dict["num_epochs"],
            eta=self.model_config_dict["eta"],
            model_loss_tolerance=self.model_config_dict["model_loss_tolerance"],
            checkpoint_path=self.model_config_dict["mlp_checkpoint_path"],
            save_checkpoint_iter=self.model_config_dict["save_checkpoint_iter"]
        )
        self.mlp_trainer.train()

    def train_pinn(self):
        print("==============Training PINN System=======================")
        self._define_model()
        self._define_optimizer()
        self._define_loss()
        self.pinn_trainer = PINN_Trainer(
            self.config_dir, self.model, self.train_loader, self.val_loader, self.test_loader,
            self.optimizer, self.criterion,
            pinn_reg_factor=self.model_config_dict["pinn_reg_factor"],
            num_epochs=self.model_config_dict["num_epochs"],
            eta=self.model_config_dict["eta"],
            model_loss_tolerance=self.model_config_dict["model_loss_tolerance"],
            checkpoint_path=self.model_config_dict["pinn_checkpoint_path"],
            save_checkpoint_iter=self.model_config_dict["save_checkpoint_iter"]
        )
        self.pinn_trainer.train()

    def train_kkt_hardnet(self):
        print("==============Training KKT-HardNet System=======================")
        self._define_model()
        self._define_optimizer()
        self._define_loss()
        self.kkt_trainer = KKT_HardNet_Trainer(
            self.config_dir, self.model, self.train_loader, self.val_loader, self.test_loader,
            self.optimizer, self.criterion,
            pinn_reg_factor=self.model_config_dict["pinn_reg_factor"],
            num_epochs=self.model_config_dict["num_epochs"],
            eta=self.model_config_dict["eta"],
            model_loss_tolerance=self.model_config_dict["model_loss_tolerance"],
            checkpoint_path=self.model_config_dict["kkt_hardnet_checkpoint_path"],
            save_checkpoint_iter=self.model_config_dict["save_checkpoint_iter"]
        )
        self.kkt_trainer.train()

    def run(self, mode="Invalid"):
        """mode can be 'mlp', 'pinn', or 'kkt'."""
        self._create_dataset()
        self._create_dataloader()

        if mode.lower() == "mlp":
            self.train_mlp()
        elif mode.lower() == "pinn":
            self.train_pinn()
        elif mode.lower() == "kkt":
            self.train_kkt_hardnet()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # pass
