import torch


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion,
                 num_epochs=500, eta=1e-3, model_loss_tolerance=1e-4, device=None):
        self.model = model
        self.res_fn = self.model.newton.res_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.eta = eta
        self.model_loss_tolerance = model_loss_tolerance
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.use_newton = False

    def train_model(self):
        self.model.train()
        total_loss = 0
        total_data_loss = 0
        total_consistency_loss = 0
        total_pinn_loss = 0

        for x_batch, y_batch, y_data_batch in self.train_loader:
            x_batch, y_batch, y_data_batch = x_batch.to(self.device), \
                                             y_batch.to(self.device), \
                                             y_data_batch.to(self.device)
            self.optimizer.zero_grad()

            y_hat = self.model.nn(x_batch)  # Predicts KKT variables (or a subset).

            # 🚫 DO NOT concatenate manually unless symbolic order demands it
            # 🔧 Construct full x_input based on symbolic parameter order
            # Assumes x_batch already contains all parameter inputs in correct order
            x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]], dim=1)

            if not self.use_newton:
                data_loss = self.criterion(y_hat[:, :y_batch.shape[-1]], y_batch)
                consistency_loss = torch.tensor(0.0, device=self.device)
                loss = data_loss  # + pinn_loss
                r = self.res_fn(y_hat, x_input)
                norm = torch.linalg.norm(r, dim=1)
                pinn_loss = torch.mean(norm)
            else:
                y_tilde = self.model.newton(y_hat, x_input)
                # print("Lambda Values: ", y_tilde[:, y_batch.shape[-1]+1:])
                data_loss = self.criterion(y_tilde[:, :y_batch.shape[-1]], y_batch)
                consistency_loss = self.criterion(y_tilde, y_hat)
                loss = data_loss  # + consistency_loss # + pinn_loss
                r = self.res_fn(y_tilde, x_input)
                norm = torch.linalg.norm(r, dim=1)
                pinn_loss = torch.mean(norm)

            loss.backward()
            self.optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_data_loss += data_loss.item() * batch_size
            total_consistency_loss += consistency_loss.item() * batch_size
            total_pinn_loss += pinn_loss.item() * batch_size

        n_samples = len(self.train_loader.dataset)
        return (total_loss / n_samples,
                total_data_loss / n_samples,
                total_consistency_loss / n_samples,
                total_pinn_loss / n_samples)

    def test_model(self):
        self.model.eval()
        total_loss = 0
        total_data_loss = 0
        total_consistency_loss = 0
        total_pinn_loss = 0

        with torch.no_grad():
            for x_test_batch, y_test_batch, y_data_test_batch in self.test_loader:
                x_test_batch, y_test_batch, y_data_test_batch = x_test_batch.to(self.device), \
                                                                y_test_batch.to(self.device), \
                                                                y_data_test_batch.to(self.device)

                y_test_hat = self.model.nn(x_test_batch)

                x_test_input = torch.cat([x_test_batch, y_test_hat[:, :y_test_batch.shape[-1]]], dim=1)

                # r = self.res_fn(y_test_hat, x_test_input)
                # norm = torch.linalg.norm(r, dim=1)
                # pinn_loss = torch.mean(norm)

                if self.use_newton:
                    # print("Lambda Values: ", y_test_hat[:, y_test_batch.shape[-1]+1:])
                    y_test_tilde = self.model.newton(y_test_hat, x_test_input)
                    r = self.res_fn(y_test_tilde, x_test_input)
                    norm = torch.linalg.norm(r, dim=1)
                    pinn_loss = torch.mean(norm)
                    data_loss = self.criterion(y_test_tilde[:, :y_test_batch.shape[-1]], y_test_batch)
                    consistency_loss = self.criterion(y_test_tilde, y_test_hat)
                    loss = data_loss  # + consistency_loss # + pinn_loss

                else:
                    r = self.res_fn(y_test_hat, x_test_input)
                    norm = torch.linalg.norm(r, dim=1)
                    pinn_loss = torch.mean(norm)
                    data_loss = self.criterion(y_test_hat[:, :y_test_batch.shape[-1]], y_test_batch)
                    consistency_loss = torch.tensor(0.0, device=self.device)
                    loss = data_loss  # + pinn_loss

                batch_size = x_test_batch.size(0)
                total_loss += loss.item() * batch_size
                total_data_loss += data_loss.item() * batch_size
                total_consistency_loss += consistency_loss.item() * batch_size
                total_pinn_loss += pinn_loss.item() * batch_size

        n_samples = len(self.test_loader.dataset)
        return (total_loss / n_samples,
                total_data_loss / n_samples,
                total_consistency_loss / n_samples,
                total_pinn_loss / n_samples)

    def display_results(self, epoch, avg_loss, avg_test_loss,
                        data_loss, consistency_loss, pinn_loss,
                        test_data_loss, test_consistency_loss, test_pinn_loss):
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Epoch {epoch + 1}]")
            print(f"  🔧 Train Total Loss = {avg_loss:.8f} | Data = {data_loss:.8f}", end='')
            # if self.use_newton:
            # print(f", Consistency = {consistency_loss:.8f}", end='')
            print(f", PINN = {pinn_loss:.8f}")

            print(f"  📊 Test  Total Loss = {avg_test_loss:.8f} | Data = {test_data_loss:.8f}", end='')
            # if self.use_newton:
            #     print(f", Consistency = {test_consistency_loss:.8f}", end='')
            print(f", PINN = {test_pinn_loss:.8f}")

    def _is_converged(self, avg_loss, avg_test_loss):
        return avg_loss < self.model_loss_tolerance and avg_test_loss < self.model_loss_tolerance

    def train(self):
        for epoch in range(self.num_epochs):
            (avg_loss, data_loss, consistency_loss, pinn_loss) = self.train_model()
            (avg_test_loss, test_data_loss, test_consistency_loss, test_pinn_loss) = self.test_model()

            self.display_results(epoch, avg_loss, avg_test_loss,
                                 data_loss, consistency_loss, pinn_loss,
                                 test_data_loss, test_consistency_loss, test_pinn_loss)

            if not self.use_newton and data_loss < self.eta:
                print(f"🔁 Activating Newton loop at epoch {epoch + 1}, loss = {avg_loss:.8f}")
                self.use_newton = True

            if self._is_converged(avg_loss, avg_test_loss):
                print(f"✅ Training complete at epoch {epoch + 1}")
                break
