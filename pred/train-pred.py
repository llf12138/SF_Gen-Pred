import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import time
import math
import numpy as np
import sys
import os


class BidirectionalLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder."""

    def __init__(self, input_size, hid_size, n_layers, dropout):
        super(BidirectionalLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hid_size, n_layers, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.n_layers = n_layers
        self.bidirectional = True

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        return self.dropout(out), hidden

class GLU(nn.Module):
    """Gated Linear Unit (GLU) activation."""

    def __init__(self, input_size, output_size=None):
        super(GLU, self).__init__()
        if output_size is None:
            output_size = input_size
        self.linear = nn.Linear(input_size, output_size * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.linear(x)
        gate, value = output.chunk(2, dim=-1)
        return value * self.sigmoid(gate)

class EnhancedFeatureExtractor(nn.Module):
    """Feature extractor with GLU gating (no attention)."""

    def __init__(self, hid_size, dropout=0.1, bidirectional=True):
        super(EnhancedFeatureExtractor, self).__init__()
        self.bidirectional = bidirectional
        feature_size = hid_size * 2 if bidirectional else hid_size

        self.glu = GLU(feature_size)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_output, hidden):
        # Global sequence representation (mean over time)
        seq_representation = torch.mean(lstm_output, dim=1)

        # Concatenate last forward/backward hidden states for BiLSTM
        if self.bidirectional:
            batch_size = hidden.size(1)
            n_layers = hidden.size(0) // 2

            hidden_reshaped = hidden.view(n_layers, 2, batch_size, -1)
            last_forward = hidden_reshaped[-1, 0]
            last_backward = hidden_reshaped[-1, 1]
            last_hidden = torch.cat([last_forward, last_backward], dim=1)
        else:
            last_hidden = hidden[-1]

        combined = seq_representation + last_hidden
        glu_output = self.glu(combined)
        # Residual connection + layer norm
        output = self.layer_norm(combined + self.dropout(glu_output))

        return output

class MLPRegressor(nn.Module):
    """MLP regression head with a uniform hidden size."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(MLPRegressor, self).__init__()

        layers = []
        prev_size = input_size

        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class pre(nn.Module):
    """BiLSTM + GLU + MLP regression model for target prediction."""

    def __init__(self, emb_size, dic_size, hid_size, n_layers, emb_dropout, dropout,
                 mlp_hidden_size=2, mlp_num_layers=3, bidirectional=True):
        super(pre, self).__init__()
        self.emb = nn.Embedding(dic_size, emb_size, padding_idx=0)
        self.drop = nn.Dropout(emb_dropout)
        self.encoder = BidirectionalLSTMEncoder(emb_size, hid_size, n_layers, dropout=dropout)

        feature_size = hid_size * 2 if bidirectional else hid_size
        self.feature_extractor = EnhancedFeatureExtractor(hid_size, dropout, bidirectional)
        self.regression_head = MLPRegressor(feature_size, mlp_hidden_size, mlp_num_layers, dropout)

        self.bidirectional = bidirectional
        self.feature_size = feature_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_num_layers = mlp_num_layers

    def forward(self, input):
        emb = self.drop(self.emb(input))
        encoded_output, hidden = self.encoder(emb)
        enhanced_features = self.feature_extractor(encoded_output, hidden)
        target_pred = self.regression_head(enhanced_features)
        return target_pred

# ---------------------------
# Utilities
# ---------------------------

def evaluate_regression(model, data_iter, criterion, device):
    """Evaluate the regression model; returns (avg loss, MAE, RMSE)."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in data_iter:
            inputs = data.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return total_loss / len(data_iter), mae, rmse

def calculate_r2(predictions, targets):
    """Compute the R^2 score."""
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))  # epsilon avoids division by zero

def print_data_info(smiles_data, target_data, name):
    """Print dataset statistics."""
    print(f"{name}: {smiles_data.shape}, target range: [{target_data.min():.3f}, {target_data.max():.3f}], "
          f"mean: {target_data.mean():.3f}, std: {target_data.std():.3f}")

# ---------------------------
# Early stopping
# ---------------------------

class EarlyStopping:
    """Stop training when the validation loss stops improving."""

    def __init__(self, patience=50, min_delta=0, verbose=True, save_path="best_model.pt"):
        """
        Args:
            patience (int): epochs to wait before stopping
            min_delta (float): minimum change counted as improvement
            verbose (bool): print early-stopping messages
            save_path (str): path of the best checkpoint
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered: no improvement for {self.patience} epochs.')
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the best model."""
        if self.verbose:
            print(f'Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        self.best_model_state = model.state_dict().copy()
        torch.save(model.state_dict(), self.save_path)

# ---------------------------
# Training and evaluation
# ---------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 128
    dropout = 0.1
    emb_dropout = 0.1
    epochs = 500
    emb_size = 512
    n_layers = 2
    lr = 0.001
    hid_size = 512
    bidirectional = True

    mlp_hidden_size = 512      # uniform hidden size across MLP layers
    mlp_num_layers = 6

    # Target selection: change this to train a different target
    target_name = "S1_exc"
    print(f"Prediction target: {target_name}")

    # Reproducibility
    torch.manual_seed(1024)
    np.random.seed(1024)

    # Load vocabulary
    vocab_path = "vocab.pt"
    vocab = torch.load(vocab_path)
    char_to_idx = vocab['char_to_idx']
    idx_to_char = vocab['idx_to_char']
    vocab_size = len(char_to_idx)
    max_len = vocab.get('max_len', None)

    print(f"Vocabulary size: {vocab_size}")
    if max_len is not None:
        print(f"Max length stored in vocab: {max_len}")
    else:
        print("Warning: max_len not found in vocab.")

    # Load the regression dataset
    regression_data = torch.load("smiles.pt")

    # Select data for the chosen target
    train_smiles = regression_data["train_smiles"]
    train_target = regression_data["train_targets"][target_name]
    val_smiles = regression_data["val_smiles"]
    val_target = regression_data["val_targets"][target_name]
    test_smiles = regression_data["test_smiles"]
    test_target = regression_data["test_targets"][target_name]

    data_padding_length = train_smiles.shape[1]
    print(f"Data padding length: {data_padding_length}")

    if max_len is not None and data_padding_length != max_len:
        print(f"Warning: data padding length ({data_padding_length}) != vocab max_len ({max_len}); "
              f"using the data padding length.")

    print_data_info(train_smiles, train_target, "Train")
    print_data_info(val_smiles, val_target, "Val")
    print_data_info(test_smiles, test_target, "Test")

    # Datasets and loaders
    train_dataset = TensorDataset(train_smiles, train_target)
    val_dataset = TensorDataset(val_smiles, val_target)
    test_dataset = TensorDataset(test_smiles, test_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data loaded.")

    # Model
    model = pre(
        emb_size=emb_size,
        dic_size=vocab_size,
        hid_size=hid_size,
        n_layers=n_layers,
        emb_dropout=emb_dropout,
        dropout=dropout,
        mlp_hidden_size=mlp_hidden_size,
        mlp_num_layers=mlp_num_layers,
        bidirectional=bidirectional
    )
    model = model.to(device)

    feature_size = hid_size * 2 if bidirectional else hid_size
    print("Regression model initialized.")
    print(f"Architecture: BiLSTM -> GLU -> MLP[{mlp_num_layers} x {mlp_hidden_size}] -> output(1)")
    print(f"Padding length: {data_padding_length}")
    print(f"Bidirectional: {model.bidirectional}")
    print(f"Feature size: {feature_size}")
    print(f"MLP config: {mlp_num_layers} hidden layers x {mlp_hidden_size} units")
    print(f"Target: {target_name}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, trainable: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    early_stopping = EarlyStopping(
        patience=50,
        min_delta=0.0001,
        verbose=True,
        save_path=f"best_model_LSTM_pre_{target_name}.pt"
    )

    best_val_loss = float('inf')
    train_history = []
    val_history = []

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            start_time = time.time()

            for data, targets in train_loader:
                inputs = data.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time

            val_loss, val_mae, val_rmse = evaluate_regression(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            train_history.append(avg_train_loss)
            val_history.append(val_loss)

            print(f'| Epoch: {epoch:3d} | train loss: {avg_train_loss:.6f} | time: {epoch_time:.2f}s')
            print(f'| val loss: {val_loss:.6f} | val MAE: {val_mae:.4f} | val RMSE: {val_rmse:.4f}')

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_LSTM_pre_{target_name}.pt")
                torch.save(model, f"model_LSTM_pre_{target_name}.pt")
                print(f'| model saved | (val loss: {val_loss:.6f})')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping: training terminated.")
                break

            print('-' * 60)

    except KeyboardInterrupt:
        print('Training interrupted by user.')

    # Final evaluation on the test set
    print('=' * 60)
    print("Evaluating the best model on the test set...")

    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        print("Loaded the best checkpoint saved by early stopping.")
    else:
        model.load_state_dict(torch.load(f"best_model_LSTM_pre_{target_name}.pt", map_location=device))
        print("Loaded the saved best model.")

    model.eval()

    test_loss, test_mae, test_rmse = evaluate_regression(model, test_loader, criterion, device)

    # R^2 on the test set
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            inputs = data.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

    r2_score = calculate_r2(np.array(all_predictions), np.array(all_targets))

    print('=' * 60)
    print(f'| Test loss (MSE): {test_loss:.6f}')
    print(f'| Test MAE: {test_mae:.4f}')
    print(f'| Test RMSE: {test_rmse:.4f}')
    print(f'| Test R^2: {r2_score:.4f}')
    print(f'| Padding length: {data_padding_length}')
    print(f'| Bidirectional: {model.bidirectional}')
    print(f'| MLP config: {mlp_num_layers} x {mlp_hidden_size}')
    print(f'| Target: {target_name}')
    print(f'| Total epochs: {epoch}')
    print(f'| Best val loss: {best_val_loss:.6f}')
    print('=' * 60)

    # Save training history
    history = {
        'train_loss': train_history,
        'val_loss': val_history,
        'test_metrics': {
            'mse': test_loss,
            'mae': test_mae,
            'rmse': test_rmse,
            'r2': r2_score
        },
        'data_padding_length': data_padding_length,
        'vocab_max_len': max_len,
        'bidirectional': bidirectional,
        'target_name': target_name,
        'model_config': {
            'emb_size': emb_size,
            'hid_size': hid_size,
            'n_layers': n_layers,
            'mlp_hidden_size': mlp_hidden_size,
            'mlp_num_layers': mlp_num_layers
        },
        'early_stopping_info': {
            'stopped_early': early_stopping.early_stop,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch
        }
    }
    torch.save(history, f"training_history_pre_{target_name}.pt")
    print("Training history saved.")

if __name__ == "__main__":
    main()
