
import torch
import numpy as np
import matplotlib.pyplot as plt


class BurnRateAnalyzer:
    def __init__(self, time_points, burn_data):
        self.time = np.array(time_points)
        self.burn_data = np.array(burn_data)
        self.log_burn_data = np.log(self.burn_data)
        self.setup_data()
        self.fit_model()

    def setup_data(self):
        """Prepare data matrices for regression"""
        self.Y = torch.from_numpy(self.log_burn_data).float().unsqueeze(-1)
        T = np.expand_dims(self.time, -1)
        X = torch.from_numpy(T).float()
        self.X = torch.cat([X, torch.ones_like(X).float()], dim=1)

    def fit_model(self):
        """Compute regression coefficients and uncertainties"""
        self.beta = torch.linalg.solve(self.X.T @ self.X, self.X.T @ self.Y)
        self.predictions = torch.matmul(self.X, self.beta)

        # Compute uncertainties
        residuals = self.Y - self.predictions
        n, p = self.X.shape
        sigma_squared = torch.sum(residuals ** 2) / (n - p)
        self.beta_covariance = sigma_squared * torch.inverse(self.X.T @ self.X)
        self.beta_std = torch.sqrt(torch.diag(self.beta_covariance))

    def plot_log_space(self):
        """Plot results in log space"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.log_burn_data)
        plt.plot(self.time, self.predictions.detach().numpy(), color='r')

        # Plot uncertainty bands
        uncertainty = self.beta_std[1].detach().numpy() + 2 * np.expand_dims(self.time, -1) * self.beta_std[
            0].detach().numpy()
        plt.plot(self.time, self.predictions.detach().numpy() + uncertainty, color='orange')
        plt.plot(self.time, self.predictions.detach().numpy() - uncertainty, color='orange')

        plt.title('Log Space - Modeled Burn Rate Prediction')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Log Cycles Burned')
        plt.legend(['data', 'mean', '2 sigma'])
        plt.tight_layout()
        plt.show()

    def plot_original_space(self):
        """Plot results in original space"""
        plt.figure(figsize=(10, 6))
        plt.bar(self.time, self.burn_data)

        # Transform predictions back to original space
        pred_orig = np.exp(self.predictions.detach().numpy())
        uncertainty = self.beta_std[1].detach().numpy() + 2 * np.expand_dims(self.time, -1) * self.beta_std[
            0].detach().numpy()

        plt.plot(self.time, pred_orig, color='r')
        plt.plot(self.time, np.exp(self.predictions.detach().numpy() + uncertainty), color='orange')
        plt.plot(self.time, np.exp(self.predictions.detach().numpy() - uncertainty), color='orange')

        plt.title('Modeled Burn Rate Prediction')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Cycles Burned')
        plt.legend(['mean', '2 sigma'])
        plt.tight_layout()
        plt.show()

    def print_results(self):
        """Print key findings"""
        rate = np.round(100 * self.beta[0].item(), 2).item()
        sigma = (self.beta[0].item() / self.beta_std[0].item())
        print(f'Exponential Rate of Increase YoY: {rate}% with {sigma:.2f}σ certainty')
