import os
import torch
import re
import time
import numpy as np
import pandas as pd


class KKT_HardNet_Trainer:
    def __init__(self, config_dir, model, train_loader, val_loader, test_loader, optimizer, criterion,
                 pinn_reg_factor=1, num_epochs=500, eta=1e-3, model_loss_tolerance=1e-4, save_checkpoint_iter=50,
                 checkpoint_path=None, device=None):
        self.config_dir = config_dir
        self.model = model
        self.sym_names = model.newton.sym_names
        self.res_fn = model.newton.res_fn
        self.eq_viol_fn = model.newton._evaluate_eq_res
        self.ineq_viol_fn = model.newton._evaluate_ineq_res
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.eta = eta
        self.pinn_reg_factor = pinn_reg_factor
        self.model_loss_tolerance = model_loss_tolerance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.use_newton = False
        self.losses_save_path = f"{self.config_dir}/kkt_hardnet_losses.npz"
        self.model_save_path = f"{self.config_dir}/kkt_hardnet_model.pth"
        self.predictions_save_path = f"{self.config_dir}/kkt_hardnet_predictions.csv"
        self.mse_mape_save_path = f"{self.config_dir}/kkt_hardnet_mse_mape.txt"

        self.train_data_losses = []
        self.test_data_losses = []
        self.train_pinn_losses = []
        self.test_pinn_losses = []
        self.train_abs_violation = []
        self.test_abs_violation = []
        self.epoch_times = []

        # # ===============================
        # # Differential Term Detection
        # # ===============================
        # self.has_differential_terms = False
        # self.required_derivatives = []
        # self.max_diff_order = None
        # diff_orders = []
        #
        # for name in self.sym_names:
        #     # Matches dy1dx1 (first-order)
        #     if re.fullmatch(r"dy\d+dx\d+", name):
        #         self.has_differential_terms = True
        #         diff_orders.append(1)
        #         y_idx, x_idx = re.findall(r"\d+", name)
        #         self.required_derivatives.append({
        #             'target': f'y{y_idx}',
        #             'order': 1,
        #             'wrt': [f'x{x_idx}'],
        #             'symbol': name
        #         })
        #
        #     # Matches d2y1dx1dx2, d3y2dx3dx1dx1, etc.
        #     elif match := re.fullmatch(r"d(\d+)y(\d+)dx(\d+)(dx\d+)*", name):
        #         self.has_differential_terms = True
        #         order = int(match.group(1))
        #         y_idx = int(match.group(2))
        #         dx_terms = re.findall(r"dx(\d+)", name)
        #         self.required_derivatives.append({
        #             'target': f'y{y_idx}',
        #             'order': order,
        #             'wrt': [f'x{i}' for i in dx_terms],
        #             'symbol': name
        #         })
        #         diff_orders.append(order)
        #
        # self.num_gradient_terms = len(self.required_derivatives)
        # if self.has_differential_terms:
        #     self.max_diff_order = max(diff_orders)

        # ===============================
        # Input/Output Variable Detection
        # ===============================
        self.input_symbols = [s for s in self.sym_names if s.startswith("x")]
        self.output_symbols = [s for s in self.sym_names if re.fullmatch(r"y\d+", s)]

        # Checkpoint detection and model weights settings:
        # If checkpoint path provided, load weights before training
        # Checkpoint Directory
        self.save_checkpoint_iter = save_checkpoint_iter
        self.best_checkpoint_loss = float('inf')

        if checkpoint_path is not None and len(checkpoint_path) == 0:
            checkpoint_path = None
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # If the checkpoint contains model_state_dict, use it
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # This means the checkpoint is just the raw state dict
                self.model.load_state_dict(checkpoint)

            # If loss is there in checkpoint
            if "loss" in checkpoint:
                print("==> Updating the best loss value to: {}".format(checkpoint["loss"]))
                self.best_checkpoint_loss = checkpoint["loss"]
                if checkpoint["loss"] < self.eta:
                    print("✅ Current Loss is less than cutoff. Using Newton from the start.")
                    self.use_newton = True

            print(f"✅ Loaded model weights from {checkpoint_path}")
        else:
            print(f"⚠️ No checkpoint file found. Starting training from scratch")

        # Make checkpoints directory
        self.checkpoint_dir = os.path.join(self.config_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, epoch, loss):
        """Save a training checkpoint if loss improves."""
        if loss < self.best_checkpoint_loss:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"kkt_hardnet_checkpoint_epoch_{epoch + 1}_loss_{loss:.6f}.pth"
            )

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            }, checkpoint_path)

            self.best_checkpoint_loss = loss
            print(f"💾 Checkpoint saved at epoch {epoch + 1} with loss {loss:.8f}")
        else:
            print(
                f"⚠️ No checkpoint saved at epoch {epoch + 1} (loss {loss:.8f} >= best {self.best_checkpoint_loss:.8f})")

    def train_model(self):
        self.model.train()
        total_loss = 0
        total_data_loss = 0
        total_consistency_loss = 0
        total_pinn_loss = 0
        total_abs_pinn_loss = 0

        for x_batch, y_batch, y_data_batch in self.train_loader:
            x_batch, y_batch, y_data_batch = (
                x_batch.to(self.device),
                y_batch.to(self.device),
                y_data_batch.to(self.device)
            )

            x_batch.requires_grad_(True)  # Enable autograd for derivative computation
            self.optimizer.zero_grad()

            # Step 1: Predict base NN outputs (y1, y2, ...)
            y_hat_base = self.model.nn(x_batch)  # shape: (B, num_outputs)

            # print("Shape of Predicted Y: ", y_hat_base.shape)

            # Step 2: Compute required derivatives
            # grad_outputs = []
            # if self.required_derivatives:
            #     output_map = {f'y{i + 1}': y_hat_base[:, i] for i in range(y_hat_base.shape[1])}

            #     for item in self.required_derivatives:
            #         y_target = output_map[item['target']]  # e.g., y1
            #         wrt_vars = item['wrt']  # e.g., ['x1', 'x2']
            #         order = item['order']

            #         # Get indices of x variables
            #         x_indices = [int(var[1:]) - 1 for var in wrt_vars]  # x1 → 0, x2 → 1, ...

            #         grad = y_target
            #         for i in range(order):
            #             grads = torch.autograd.grad(
            #                 grad,
            #                 x_test_batch,
            #                 grad_outputs=torch.ones_like(grad),
            #                 create_graph=True,
            #                 retain_graph=True,
            #                 only_inputs=True,
            #                 allow_unused=True
            #             )[0]

            #             if grads is None:
            #                 grads = torch.zeros_like(x_test_batch)

            #             grad = grads[:, x_indices[i]]

            #         grad_outputs.append(grad.unsqueeze(1))  # shape: (B, 1)

            # grad_outputs = []
            # if self.required_derivatives:
            #     # map output names to columns
            #     output_map = {f'y{i + 1}': y_hat_base[:, i] for i in range(y_hat_base.shape[1])}
            #
            #     for item in self.required_derivatives:
            #         target_name = item['target']  # e.g., 'y1'
            #         order = int(item['order'])  # integer order
            #         wrt_list = item.get('wrt', [])  # e.g., ['x1'] or ['x1','x2']
            #         y_target = output_map[target_name]  # shape [B]
            #
            #         # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
            #         wrt_indices = [int(v[1:]) - 1 for v in wrt_list]
            #
            #         # build differentiation sequence (length == order)
            #         # - if user provided exactly 'order' vars, use them
            #         # - if single entry provided, repeat it 'order' times
            #         # - if fewer than order provided, repeat the last one to fill
            #         if len(wrt_indices) == order:
            #             seq = wrt_indices
            #         elif len(wrt_indices) == 1:
            #             seq = [wrt_indices[0]] * order
            #         elif 1 < len(wrt_indices) < order:
            #             seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
            #         else:  # if none provided, default to differentiate w.r.t. x1 repeatedly
            #             seq = [0] * order
            #
            #         # iterative differentiation following the sequence
            #         grad = y_target  # shape [B]
            #         # we need a scalar-valued grad_outputs of same shape for autograd.grad
            #         for j, var_idx in enumerate(seq):
            #             # compute gradients of 'grad' w.r.t. x_batch (returns [B, input_dim])
            #             grads_wrt_x = torch.autograd.grad(
            #                 grad,
            #                 x_batch,
            #                 grad_outputs=torch.ones_like(grad),
            #                 create_graph=True,
            #                 retain_graph=True,
            #                 only_inputs=True
            #             )[0]  # shape [B, input_dim]
            #
            #             # select the column corresponding to var_idx
            #             grad = grads_wrt_x[:, var_idx]  # shape [B]
            #
            #         # now 'grad' is the desired derivative array shape [B]
            #         grad_outputs.append(grad)  # collect for later concatenation/use
            #
            # # # Step 3: Concatenate outputs according to sym_names
            # # y_head = y_hat_base[:, :y_test_batch.shape[-1]]  # original y vars
            # # y_tail = y_hat_base[:, y_test_batch.shape[-1]:]  # remaining vars after y
            #
            # # if grad_outputs:
            # #     y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
            # # else:
            # #     y_hat = y_hat_base
            #
            # # Ensure all grad_outputs are 2D column tensors [B, 1]
            # grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]
            #
            # # Step 3: Concatenate all outputs in the order of sym_names
            # y_head = y_hat_base[:, :y_batch.shape[-1]]  # original y vars
            # y_tail = y_hat_base[:, y_batch.shape[-1]:]  # remaining vars after y
            #
            # if grad_outputs:
            #     y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
            # else:
            y_hat = y_hat_base

            # Step 4: Build input for res_fn (x + original NN outputs)
            x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]], dim=1)

            if not self.use_newton:
                data_loss = self.criterion(y_hat[:, :y_batch.shape[-1]], y_batch)
                consistency_loss = torch.tensor(0.0, device=self.device)
                loss = data_loss
                # r = self.res_fn(y_hat, x_input)
                # norm = torch.linalg.norm(r, dim=1)
                # pinn_loss = torch.mean(norm)

                eq_res = self.eq_viol_fn(y_hat, x_input)
                ineq_res = self.ineq_viol_fn(y_hat, x_input)
                # eq = eq_res if eq_res.dim() == 2 else eq_res.squeeze(-1)
                # iq = ineq_res if ineq_res.dim() == 2 else ineq_res.squeeze(-1)
                # iq_viol = torch.clamp_min(iq, 0.0)
                # combined = torch.cat([eq, iq_viol], dim=1)
                ineq_res = torch.clamp_min(ineq_res, 0)
                combined = torch.cat([eq_res, ineq_res], dim=1)
                abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
                pinn_loss = torch.linalg.norm(combined, dim=1).mean()

            else:
                # print("Total Y_hat Input to Newton: ", y_hat)
                y_tilde = self.model.newton(y_hat, x_input)
                # print("Total Y_tilde Output from Newton: ", y_tilde)
                data_loss = self.criterion(y_tilde[:, :y_batch.shape[-1]], y_batch)
                consistency_loss = self.criterion(y_tilde, y_hat)
                loss = data_loss
                # r = self.res_fn(y_tilde, x_input)
                # norm = torch.linalg.norm(r, dim=1)
                # pinn_loss = torch.mean(norm)

                eq_res = self.eq_viol_fn(y_tilde, x_input)
                ineq_res = self.ineq_viol_fn(y_tilde, x_input)
                # eq = eq_res if eq_res.dim() == 2 else eq_res.squeeze(-1)
                # iq = ineq_res if ineq_res.dim() == 2 else ineq_res.squeeze(-1)
                # iq_viol = torch.clamp_min(iq, 0.0)
                # combined = torch.cat([eq, iq_viol], dim=1)
                ineq_res = torch.clamp_min(ineq_res, 0)
                combined = torch.cat([eq_res, ineq_res], dim=1)
                abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
                pinn_loss = torch.linalg.norm(combined, dim=1).mean()

            # Step 5: Backpropagation and optimizer step
            loss.backward()
            self.optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_data_loss += data_loss.item() * batch_size
            total_consistency_loss += consistency_loss.item() * batch_size
            total_pinn_loss += pinn_loss.item() * batch_size
            total_abs_pinn_loss += abs_pinn_loss.item() * batch_size

        n_samples = len(self.train_loader.dataset)
        return (
            total_loss / n_samples,
            total_data_loss / n_samples,
            total_consistency_loss / n_samples,
            total_pinn_loss / n_samples,
            total_abs_pinn_loss / n_samples
        )

    def test_model(self):
        self.model.eval()
        total_loss = 0
        total_data_loss = 0
        total_consistency_loss = 0
        total_pinn_loss = 0
        total_abs_pinn_loss = 0

        for x_test_batch, y_test_batch, y_data_test_batch in self.test_loader:
            x_test_batch = x_test_batch.to(self.device).requires_grad_(True)
            y_test_batch = y_test_batch.to(self.device)
            y_data_test_batch = y_data_test_batch.to(self.device)

            # Step 1: Base NN predictions
            y_hat_base = self.model.nn(x_test_batch)  # shape: (B, num_outputs)

            # Step 2: Compute required derivatives
            # grad_outputs = []
            # if self.required_derivatives:
            #     output_map = {f'y{i + 1}': y_hat_base[:, i] for i in range(y_hat_base.shape[1])}

            #     for item in self.required_derivatives:
            #         y_target = output_map[item['target']]  # e.g., y1
            #         wrt_vars = item['wrt']  # e.g., ['x1', 'x2']
            #         order = item['order']

            #         # Get indices of x variables
            #         x_indices = [int(var[1:]) - 1 for var in wrt_vars]  # x1 → 0, x2 → 1, ...

            #         grad = y_target
            #         for i in range(order):
            #             grads = torch.autograd.grad(
            #                 grad,
            #                 x_test_batch,
            #                 grad_outputs=torch.ones_like(grad),
            #                 create_graph=True,
            #                 retain_graph=True,
            #                 only_inputs=True,
            #                 allow_unused=True
            #             )[0]

            #             if grads is None:
            #                 grads = torch.zeros_like(x_test_batch)

            #             grad = grads[:, x_indices[i]]

            #         grad_outputs.append(grad.unsqueeze(1))  # shape: (B, 1)

            # grad_outputs = []
            # if self.required_derivatives:
            #     # map output names to columns
            #     output_map = {f'y{i + 1}': y_hat_base[:, i] for i in range(y_hat_base.shape[1])}
            #
            #     for item in self.required_derivatives:
            #         target_name = item['target']  # e.g., 'y1'
            #         order = int(item['order'])  # integer order
            #         wrt_list = item.get('wrt', [])  # e.g., ['x1'] or ['x1','x2']
            #         y_target = output_map[target_name]  # shape [B]
            #
            #         # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
            #         wrt_indices = [int(v[1:]) - 1 for v in wrt_list]
            #
            #         # build differentiation sequence (length == order)
            #         # - if user provided exactly 'order' vars, use them
            #         # - if single entry provided, repeat it 'order' times
            #         # - if fewer than order provided, repeat the last one to fill
            #         if len(wrt_indices) == order:
            #             seq = wrt_indices
            #         elif len(wrt_indices) == 1:
            #             seq = [wrt_indices[0]] * order
            #         elif 1 < len(wrt_indices) < order:
            #             seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
            #         else:  # if none provided, default to differentiate w.r.t. x1 repeatedly
            #             seq = [0] * order
            #
            #         # iterative differentiation following the sequence
            #         grad = y_target  # shape [B]
            #         # we need a scalar-valued grad_outputs of same shape for autograd.grad
            #         for j, var_idx in enumerate(seq):
            #             # compute gradients of 'grad' w.r.t. x_batch (returns [B, input_dim])
            #             grads_wrt_x = torch.autograd.grad(
            #                 grad,
            #                 x_test_batch,
            #                 grad_outputs=torch.ones_like(grad),
            #                 create_graph=True,
            #                 retain_graph=True,
            #                 only_inputs=True
            #             )[0]  # shape [B, input_dim]
            #
            #             # select the column corresponding to var_idx
            #             grad = grads_wrt_x[:, var_idx]  # shape [B]
            #
            #         # now 'grad' is the desired derivative array shape [B]
            #         grad_outputs.append(grad)  # collect for later concatenation/use
            #
            # # # Step 3: Concatenate outputs according to sym_names
            # # y_head = y_hat_base[:, :y_test_batch.shape[-1]]  # original y vars
            # # y_tail = y_hat_base[:, y_test_batch.shape[-1]:]  # remaining vars after y
            #
            # # if grad_outputs:
            # #     y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
            # # else:
            # #     y_hat = y_hat_base
            #
            # # Ensure all grad_outputs are 2D column tensors [B, 1]
            # grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]
            #
            # # Step 3: Concatenate all outputs in the order of sym_names
            # y_head = y_hat_base[:, :y_test_batch.shape[-1]]  # original y vars
            # y_tail = y_hat_base[:, y_test_batch.shape[-1]:]  # remaining vars after y
            #
            # if grad_outputs:
            #     y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
            # else:
            y_hat = y_hat_base

            # Step 4: Prepare residual input
            x_input = torch.cat([x_test_batch, y_hat[:, :y_test_batch.shape[-1]]], dim=1)

            # Step 5: Evaluate losses
            if self.use_newton:
                y_tilde = self.model.newton(y_hat, x_input)
                # r = self.res_fn(y_tilde, x_input)
                # norm = torch.linalg.norm(r, dim=1)
                # pinn_loss = torch.mean(norm)

                eq_res = self.eq_viol_fn(y_tilde, x_input)
                ineq_res = self.ineq_viol_fn(y_tilde, x_input)
                # eq = eq_res if eq_res.dim() == 2 else eq_res.squeeze(-1)
                # iq = ineq_res if ineq_res.dim() == 2 else ineq_res.squeeze(-1)
                # iq_viol = torch.clamp_min(iq, 0.0)
                # combined = torch.cat([eq, iq_viol], dim=1)
                ineq_res = torch.clamp_min(ineq_res, 0)
                combined = torch.cat([eq_res, ineq_res], dim=1)
                abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
                pinn_loss = torch.linalg.norm(combined, dim=1).mean()

                data_loss = self.criterion(y_tilde[:, :y_test_batch.shape[-1]], y_test_batch)
                consistency_loss = self.criterion(y_tilde, y_hat)
                loss = data_loss
            else:
                # r = self.res_fn(y_hat, x_input)
                # norm = torch.linalg.norm(r, dim=1)
                # pinn_loss = torch.mean(norm)

                eq_res = self.eq_viol_fn(y_hat, x_input)
                ineq_res = self.ineq_viol_fn(y_hat, x_input)
                # eq = eq_res if eq_res.dim() == 2 else eq_res.squeeze(-1)
                # iq = ineq_res if ineq_res.dim() == 2 else ineq_res.squeeze(-1)
                # iq_viol = torch.clamp_min(iq, 0.0)
                # combined = torch.cat([eq, iq_viol], dim=1)
                ineq_res = torch.clamp_min(ineq_res, 0)
                combined = torch.cat([eq_res, ineq_res], dim=1)
                abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
                pinn_loss = torch.linalg.norm(combined, dim=1).mean()

                data_loss = self.criterion(y_hat[:, :y_test_batch.shape[-1]], y_test_batch)
                consistency_loss = torch.tensor(0.0, device=self.device)
                loss = data_loss

            batch_size = x_test_batch.size(0)
            total_loss += loss.item() * batch_size
            total_data_loss += data_loss.item() * batch_size
            total_consistency_loss += consistency_loss.item() * batch_size
            total_pinn_loss += pinn_loss.item() * batch_size
            total_abs_pinn_loss += abs_pinn_loss.item() * batch_size

        n_samples = len(self.test_loader.dataset)
        return (
            total_loss / n_samples,
            total_data_loss / n_samples,
            total_consistency_loss / n_samples,
            total_pinn_loss / n_samples,
            total_abs_pinn_loss / n_samples
        )

    def display_results(self, epoch, avg_loss, avg_test_loss,
                        data_loss, consistency_loss, pinn_loss, abs_violation,
                        test_data_loss, test_consistency_loss, test_pinn_loss, test_abs_violation):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch + 1}]")
            print(f"  🔧 Train Total Loss = {avg_loss:.8f} | Data = {data_loss:.8f}", end='')
            # if self.use_newton:
            # print(f", Consistency = {consistency_loss:.8f}", end='')
            print(f", Physics = {pinn_loss:.8f}", end='')
            print(f", Absolute Violation = {abs_violation:.8f}")

            print(f"  📊 Test  Total Loss = {avg_test_loss:.8f} | Data = {test_data_loss:.8f}", end='')
            # if self.use_newton:
            #     print(f", Consistency = {test_consistency_loss:.8f}", end='')
            print(f", Physics = {test_pinn_loss:.8f}", end='')
            print(f", Absolute Violation = {test_abs_violation:.8f}")

    def _is_converged(self, avg_loss, avg_test_loss):
        return avg_loss < self.model_loss_tolerance and avg_test_loss < self.model_loss_tolerance

    def _save_losses(self):
        np.savez(self.losses_save_path,
                 train_data_loss=np.array(self.train_data_losses),
                 train_pinn_loss=np.array(self.train_pinn_losses),
                 train_abs_violation=np.array(self.train_abs_violation),
                 test_data_loss=np.array(self.test_data_losses),
                 test_pinn_loss=np.array(self.test_pinn_losses),
                 test_abs_violation=np.array(self.test_abs_violation),
                 epoch_time=np.array(self.epoch_times))

    def _save_model(self):
        torch.save(self.model.nn.state_dict(), self.model_save_path)

    def export_predictions(self):
        self.model.eval()
        x_all, y_all = [], []

        for loader in [self.train_loader, self.test_loader]:
            for x_batch, y_batch, *_ in loader:
                x_batch = x_batch.to(self.device).requires_grad_(True)
                y_batch = y_batch.to(self.device)

                # Step 1: Base NN predictions
                y_hat = self.model.nn(x_batch)

                # Step 2: Compute required derivatives
                # grad_outputs = []
                # if self.required_derivatives:
                #     output_map = {f'y{i + 1}': y_hat[:, i] for i in range(y_hat.shape[1])}
                #
                #     for item in self.required_derivatives:
                #         target_name = item['target']
                #         order = int(item['order'])
                #         wrt_list = item.get('wrt', [])
                #         y_target = output_map[target_name]
                #
                #         # convert wrt variable names to indices
                #         wrt_indices = [int(v[1:]) - 1 for v in wrt_list]
                #
                #         # build differentiation sequence
                #         if len(wrt_indices) == order:
                #             seq = wrt_indices
                #         elif len(wrt_indices) == 1:
                #             seq = [wrt_indices[0]] * order
                #         elif 1 < len(wrt_indices) < order:
                #             seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                #         else:
                #             seq = [0] * order
                #
                #         grad = y_target
                #         for var_idx in seq:
                #             grads_wrt_x = torch.autograd.grad(
                #                 grad,
                #                 x_batch,
                #                 grad_outputs=torch.ones_like(grad),
                #                 create_graph=True,
                #                 retain_graph=True,
                #                 only_inputs=True
                #             )[0]
                #             grad = grads_wrt_x[:, var_idx]
                #
                #         grad_outputs.append(grad)
                #
                # # Ensure all grad_outputs are 2D column tensors [B, 1]
                # grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]
                #
                # # Step 3: Concatenate outputs in the order of sym_names
                # y_head = y_hat[:, :y_batch.shape[-1]]
                # y_tail = y_hat[:, y_batch.shape[-1]:]
                #
                # if grad_outputs:
                #     y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)

                # Step 4: Prepare input for Newton
                x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]], dim=1)
                y_tilde = self.model.newton(y_hat, x_input)

                # Step 5: Save predictions
                y_pred = y_tilde[:, :y_batch.shape[-1]].detach().cpu().numpy()
                x_np = x_batch.detach().cpu().numpy()

                x_all.append(x_np)
                y_all.append(y_pred)

        # Combine all data
        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)

        # Create DataFrame and save
        import pandas as pd
        df = pd.DataFrame(
            np.hstack([x_all, y_all]),
            columns=[f"x{i + 1}" for i in range(x_all.shape[1])] +
                    [f"y{i + 1}" for i in range(y_all.shape[1])]
        )

        df.to_csv(self.predictions_save_path, index=False)
        print(f"💾 Predictions saved at {self.predictions_save_path}")

    def mse_mape(self, loader):
        self.model.eval()
        all_abs_violation = torch.empty(0, device=self.device)
        all_norm_violation = torch.empty(0, device=self.device)
        y_data = []
        y_pred = []

        for x_batch, y_batch, *_ in loader:
            x_batch = x_batch.to(self.device).requires_grad_(True)
            y_batch = y_batch.to(self.device)

            # Step 1: Base NN predictions
            y_hat = self.model.nn(x_batch)

            # # Step 2: Compute required derivatives
            # grad_outputs = []
            # if self.required_derivatives:
            #     output_map = {f'y{i + 1}': y_hat[:, i] for i in range(y_hat.shape[1])}
            #
            #     for item in self.required_derivatives:
            #         target_name = item['target']
            #         order = int(item['order'])
            #         wrt_list = item.get('wrt', [])
            #         y_target = output_map[target_name]
            #
            #         # convert wrt variable names to indices
            #         wrt_indices = [int(v[1:]) - 1 for v in wrt_list]
            #
            #         # build differentiation sequence
            #         if len(wrt_indices) == order:
            #             seq = wrt_indices
            #         elif len(wrt_indices) == 1:
            #             seq = [wrt_indices[0]] * order
            #         elif 1 < len(wrt_indices) < order:
            #             seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
            #         else:
            #             seq = [0] * order
            #
            #         grad = y_target
            #         for var_idx in seq:
            #             grads_wrt_x = torch.autograd.grad(
            #                 grad,
            #                 x_batch,
            #                 grad_outputs=torch.ones_like(grad),
            #                 create_graph=True,
            #                 retain_graph=True,
            #                 only_inputs=True
            #             )[0]
            #             grad = grads_wrt_x[:, var_idx]
            #
            #         grad_outputs.append(grad)
            #
            # # Ensure all grad_outputs are 2D column tensors [B, 1]
            # grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]
            #
            # # Step 3: Concatenate outputs in the order of sym_names
            # y_head = y_hat[:, :y_batch.shape[-1]]
            # y_tail = y_hat[:, y_batch.shape[-1]:]
            #
            # if grad_outputs:
            #     y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)

            # Step 4: Prepare input for Newton
            x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]], dim=1)
            y_tilde = self.model.newton(y_hat, x_input)

            # Step 5: Save predictions
            y_batch_pred = y_tilde[:, :y_batch.shape[-1]]

            # Compute Constraint Violation
            x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]], dim=1)
            eq_res = self.eq_viol_fn(y_tilde, x_input)
            ineq_res = self.ineq_viol_fn(y_tilde, x_input)
            ineq_res = torch.clamp_min(ineq_res, 0)
            combined = torch.cat([eq_res, ineq_res], dim=1)
            abs_violation = combined.abs().sum(dim=1)

            all_abs_violation = torch.cat([all_abs_violation, abs_violation], dim=0)
            all_norm_violation = torch.cat([all_norm_violation, combined])

            y_data.append(y_batch)
            y_pred.append(y_batch_pred)

        y_data_all = torch.cat(y_data, dim=0)
        y_pred_all = torch.cat(y_pred, dim=0)

        mse = torch.mean((y_pred_all - y_data_all) ** 2).item()
        mape = torch.mean((y_pred_all - y_data_all).abs() / y_data_all.abs().clamp_min(1e-12)).item()
        average_abs_violation = torch.mean(all_abs_violation).item()
        average_norm_violation = torch.linalg.norm(all_norm_violation, dim=1).mean()

        return mse, mape, average_abs_violation, average_norm_violation

    def _save_mse_mape(self):
        mse_train, mape_train, abs_violation_train, norm_violation_train = self.mse_mape(loader=self.train_loader)
        mse_test, mape_test, abs_violation_test, norm_violation_test = self.mse_mape(loader=self.test_loader)

        with open(self.mse_mape_save_path, "w") as f:
            f.write(f"==================== Train Metrices ====================\n")
            f.write(f"mse train = {mse_train}\n")
            f.write(f"mape train = {mape_train}\n")
            f.write(f"absolute violation train = {abs_violation_train}\n")
            f.write(f"norm violation train = {norm_violation_train}\n\n")
            f.write(f"==================== Test Metrices =====================\n")
            f.write(f"mse test = {mse_test}\n")
            f.write(f"mape test = {mape_test}\n")
            f.write(f"absolute violation test = {abs_violation_test}\n")
            f.write(f"norm violation test = {norm_violation_test}")
            print(f"💾 MSE and MAPE saved at file {self.mse_mape_save_path}")

    def train(self):
        # Create lists to store the losses

        for epoch in range(self.num_epochs):
            start_time = time.time()
            (avg_loss, data_loss, consistency_loss, pinn_loss, abs_violation) = self.train_model()
            (avg_test_loss, test_data_loss, test_consistency_loss, test_pinn_loss,
             test_abs_violation) = self.test_model()
            end_time = time.time()
            epoch_duration = end_time - start_time
            self.epoch_times.append(epoch_duration)

            # Store losses
            self.train_data_losses.append(data_loss)
            self.train_pinn_losses.append(pinn_loss)
            self.train_abs_violation.append(abs_violation)
            self.test_data_losses.append(test_data_loss)
            self.test_pinn_losses.append(test_pinn_loss)
            self.test_abs_violation.append(test_abs_violation)

            self.display_results(epoch, avg_loss, avg_test_loss,
                                 data_loss, consistency_loss, pinn_loss, abs_violation,
                                 test_data_loss, test_consistency_loss, test_pinn_loss, test_abs_violation)

            # Save checkpoint every save_checkpoint_iter if loss improves
            if (epoch + 1) % self.save_checkpoint_iter == 0:
                self._save_checkpoint(epoch, avg_test_loss)

            if not self.use_newton and data_loss < self.eta:
                print(f"🔁 Activating Newton loop at epoch {epoch + 1}, loss = {avg_loss:.8f}")
                self.use_newton = True

            if self._is_converged(avg_loss, avg_test_loss):
                print(f"✅ Training complete at epoch {epoch + 1}")
                break

        with open(self.mse_mape_save_path, "w") as f:
            f.write(f"==================== Train Metrices ====================\n")
            f.write(f"mse train = {data_loss}\n")
            f.write(f"absolute violation train = {abs_violation}\n")
            f.write(f"norm violation train = {pinn_loss}\n\n")
            f.write(f"==================== Test Metrices =====================\n")
            f.write(f"mse test = {test_data_loss}\n")
            f.write(f"absolute violation test = {test_abs_violation}\n")
            f.write(f"norm violation test = {test_pinn_loss}")
            print(f"💾 MSE and MAPE saved at file {self.mse_mape_save_path}")

        self._save_model()
        self._save_losses()
        self.export_predictions()
        # self._save_mse_mape()
