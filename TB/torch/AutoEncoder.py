import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from matplotlib.pyplot import scatter, title, show

# Check if CUDA (GPU) is available
print(f"CUDA is available: {torch.cuda.is_available()}")
# Print the current CUDA device (if available)
print(
    f"CUDA current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name()}"
)


class AutoEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            latent_dim=2,
            layers=[64, 32],
            add_sigmoid=True,
            dropout=0,
            device=None,
            verbose=False,
            save_every=None,
            show_plots=False,
            random_state=None,
    ):
        """
        PyTorch based autoencoder providing a scikit-learn API.

        Args:
            - input_dim: int, number of input and output features
                for the neural network.
            - latent_dim: int, dimension of the latent or encoded space.
            - layers: Array(int), defines the architecture of the
                hidden layers.
            - add_sigmoid: bool, whether or not to add a sigmoid
                transformation at the middle layer.
            - dropout: float, (0-1), dropout factor before each
                layer in the encoder.
            - device: CUDA device to use (if available).
            - verbose: bool, whether to print network architecture and CUDA info.
            - save_every: int, save intermediate snapshots every n epochs.
            - show_plots: bool, display scatter plots during training.
            - random_state: Seed for random number generation (for reproducibility).
        """

        if random_state is not None:
            torch.manual_seed(random_state)

        super(AutoEncoder, self).__init()

        self.save_every = save_every
        self.show_plots = show_plots
        self.snapshots = []

        layout = [input_dim] + layers + [latent_dim]
        layout = [(i, j) for i, j in zip(layout[:-1], layout[1:])]

        encoder = []

        for i, j in layout:
            encoder.append(nn.Dropout(dropout))
            encoder.append(nn.Linear(i, j))
            encoder.append(nn.ReLU())
        encoder.pop()

        if add_sigmoid:
            encoder.append(nn.Sigmoid())

        layout.reverse()

        decoder = []
        for j, i in layout:
            decoder.append(nn.Linear(i, j))
            decoder.append(nn.ReLU())
        decoder.pop()

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.snapshots = []
        self.device = device
        self.to(device)
        self._epoch = 0

        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)

        if verbose:
            for i in encoder:
                print(i)
            for i in decoder:
                print(i)
            if torch.cuda.is_available():
                print("Cuda available")
            else:
                print("CUDA is not available")
            print(f"Using device {current_device} ({current_device_name})")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(
            self,
            X_train,
            epochs,
            batch_size=8,
            labels=None,
            num_workers=4,
            shuffle=True,
            hue=None,
            lr=1e-4,
    ):
        """
        Fitting function to train the neural network.

        Args:
            - X_train: Training data (pd.DataFrame or np.array).
            - epochs: Number of training epochs.
            - batch_size: Batch size for training.
            - labels: Labels for data (if available).
            - num_workers: Number of workers for data loading.
            - shuffle: Whether to shuffle data during training.
            - hue: Hue for visualization (if available).
            - lr: Learning rate for optimization.
        """

        if isinstance(X_train, pd.DataFrame):
            ndx = X_train.index
            X_train = X_train.values
            X_train = torch.tensor(X_train).float()

        dataloader = DataLoader(
            X_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        optimizer = optim.Adam(self.parameters(), lr=lr)

        criterion = nn.MSELoss()

        for i in tqdm(range(1, epochs + 1)):
            self._epoch += 1
            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                # Compute reconstructions
                outputs = self(data)
                # Compute training reconstruction loss
                train_loss = criterion(outputs, data)
                # Compute accumulated gradients
                train_loss.backward()
                # Perform parameter update based on current gradients
                optimizer.step()
                # Compute the epoch training loss
                assert train_loss is not np.NaN

            if (self.save_every is not None) and ((i) % self.save_every == 0):
                result = pd.DataFrame(
                    self.encoder(X_train.to(self.device)).detach().cpu().numpy(),
                    index=ndx,
                )

                result.columns = [c + 1 for c in result.columns]
                result = result.add_prefix("AE-")
                result["Epoch"] = self._epoch
                result["Labels"] = labels
                result.index = ndx
                self.snapshots.append(result)
                if self.show_plots:
                    sns.relplot(
                        data=result,
                        x="AE-0",
                        y="AE-1",
                        hue="Labels",
                        kind="scatter",
                        height=3,
                    )
                    title(f"Epoch {i}, loss={train_loss:2.2f}")
                    show()

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            ndx = X.index
            X = X.values
        else:
            ndx = None
        X = torch.tensor(X).float().to(self.device)
        enc = self.encoder(X)
        enc = pd.DataFrame(enc)
        enc.columns = [c + 1 for c in enc.columns]
        enc = enc.add_prefix("AE-").astype(float)
        if ndx is not None:
            enc.index = ndx
        return enc

    def plot(self, **kwargs):
        sn = pd.concat(self.snapshots).reset_index()
        display(sn)
        fig = px.scatter(sn, x='AE-1', y='AE-2', animation_frame='Epoch', **kwargs)
        fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
        return fig
