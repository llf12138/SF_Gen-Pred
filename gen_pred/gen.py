import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
import time
import os
import argparse
import pandas as pd
import numpy as np
import csv
import gc

RDLogger.DisableLog('rdApp.*')

# ---------------------------
# Generative model
# ---------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hid_size, n_layers, dropout):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hid_size, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out)


class GEN(nn.Module):
    def __init__(self, emb_size, dic_size, hid_size, n_layers, emb_dropout, dropout):
        super(GEN, self).__init__()
        self.emb = nn.Embedding(dic_size, emb_size, padding_idx=0)
        self.drop = nn.Dropout(emb_dropout)
        self.encoder = LSTMEncoder(emb_size, hid_size, n_layers, dropout=dropout)
        self.decoder = nn.Linear(hid_size, dic_size)

    def forward(self, input):
        emb = self.drop(self.emb(input))
        y = self.encoder(emb)
        o = self.decoder(y)
        return o.contiguous()


class Generator:
    def __init__(self, model_path, vocab_path, device):
        self.device = device
        vocab = torch.load(vocab_path, map_location=self.device)
        self.char_to_idx = vocab['char_to_idx']
        self.idx_to_char = vocab['idx_to_char']
        self.vocab_size = len(self.char_to_idx)
        self.max_length = vocab['max_len']

        self.model = torch.load(model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_batch(self, batch_size, temperature):
        seed_idx = self.char_to_idx['?']
        input_seq = torch.full((batch_size, 1), seed_idx, device=self.device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        sequences = [['?'] for _ in range(batch_size)]

        for _ in range(self.max_length):
            with torch.no_grad():
                output = self.model(input_seq)
            output = output[:, -1, :] / temperature
            probabilities = torch.softmax(output, dim=-1)

            next_idxs = torch.multinomial(probabilities, 1).view(-1)

            for i in range(batch_size):
                if not finished[i]:
                    char = self.idx_to_char[next_idxs[i].item()]
                    sequences[i].append(char)
                    if char == '+':
                        finished[i] = True

            if finished.all():
                break

            input_seq = torch.cat([input_seq, next_idxs.unsqueeze(1)], dim=1)

        return [''.join(seq).replace('?', '').replace('+', '') for seq in sequences]


# ---------------------------
# Prediction model
# ---------------------------
class BidirectionalLSTMEncoder(nn.Module):
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
    def __init__(self, hid_size, dropout=0.1, bidirectional=True):
        super(EnhancedFeatureExtractor, self).__init__()
        self.bidirectional = bidirectional
        feature_size = hid_size * 2 if bidirectional else hid_size
        self.glu = GLU(feature_size)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_output, hidden):
        seq_representation = torch.mean(lstm_output, dim=1)
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
        output = self.layer_norm(combined + self.dropout(glu_output))
        return output


class MLPRegressor(nn.Module):
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


class pre(nn.Module):  # class name must match the saved checkpoints
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
def is_valid_smiles(smiles):
    """Check SMILES validity with RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def encode_smiles(smiles, vocab_dict, max_len):
    """Encode a SMILES string with the predictor vocabulary; return None on unknown tokens or overflow."""
    try:
        encoded = [vocab_dict["?"]]  # start symbol
        i = 0
        while i < len(smiles):
            if smiles[i] == "[":
                j = smiles.find("]", i)
                if j != -1:
                    token = smiles[i: j + 1]
                    if token not in vocab_dict:
                        return None
                    encoded.append(vocab_dict[token])
                    i = j + 1
                else:
                    return None
            elif i + 1 < len(smiles) and smiles[i: i + 2] in vocab_dict:
                token = smiles[i: i + 2]
                encoded.append(vocab_dict[token])
                i += 2
            else:
                token = smiles[i]
                if token not in vocab_dict:
                    return None
                encoded.append(vocab_dict[token])
                i += 1
        while len(encoded) < max_len:
            encoded.append(0)  # padding index 0
        if len(encoded) > max_len:
            return None
        return encoded
    except Exception:
        return None


class Predictor:
    """Loads the prediction models and runs batch inference."""

    def __init__(self, model_paths, vocab_path, target_names, device):
        self.device = device
        self.target_names = target_names
        vocab = torch.load(vocab_path, map_location='cpu')
        self.char_to_idx = vocab['char_to_idx']
        self.max_len = vocab['max_len']

        self.models = {}
        for name in target_names:
            model = torch.load(model_paths[name], map_location=device)
            model.to(device)
            model.eval()
            self.models[name] = model
            print(f"Loaded prediction model for {name}")

    def predict_smiles_batch(self, smiles_list):
        """Predict a batch of SMILES; returns ({target: np.array}, valid_indices)."""
        encoded_tensors = []
        valid_indices = []
        for idx, smi in enumerate(smiles_list):
            enc = encode_smiles(smi, self.char_to_idx, self.max_len)
            if enc is not None:
                encoded_tensors.append(torch.tensor(enc, dtype=torch.long))
                valid_indices.append(idx)
        if not encoded_tensors:
            return {}, []

        batch_tensor = torch.stack(encoded_tensors).to(self.device)

        predictions = {}
        with torch.no_grad():
            for name in self.target_names:
                pred = self.models[name](batch_tensor)
                predictions[name] = pred.cpu().numpy()

        del batch_tensor
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return predictions, valid_indices


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate SMILES and predict three excitation energies')
    parser.add_argument('--num_smiles', type=int, default=64,
                        help='number of valid SMILES to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for generation')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='sampling temperature')
    parser.add_argument('--output', type=str, default='generated_predictions.csv',
                        help='output CSV path')
    parser.add_argument('--gen_model', type=str, default='gen/model_gen.pt',
                        help='generator model path')
    parser.add_argument('--gen_vocab', type=str, default='gen/vocab.pt',
                        help='generator vocabulary path')
    parser.add_argument('--pred_model_dir', type=str, default='pred',
                        help='directory of prediction models')
    parser.add_argument('--pred_vocab', type=str, default='pred/vocab.pt',
                        help='predictor vocabulary path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading generator...")
    generator = Generator(args.gen_model, args.gen_vocab, device)
    print(f"Generator max length: {generator.max_length}")

    print("Loading predictor...")
    target_names = ["S1_exc", "T1_exc", "T2_exc"]
    model_paths = {
        "S1_exc": os.path.join(args.pred_model_dir, "model_LSTM_pre_S1_exc.pt"),
        "T1_exc": os.path.join(args.pred_model_dir, "model_LSTM_pre_T1_exc.pt"),
        "T2_exc": os.path.join(args.pred_model_dir, "model_LSTM_pre_T2_exc.pt"),
    }
    predictor = Predictor(model_paths, args.pred_vocab, target_names, device)

    # Temporary CSV for generated SMILES
    temp_csv = "temp_generated_smiles.csv"
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    print(f"Generating {args.num_smiles} valid SMILES...")
    total_generated = 0
    valid_count = 0
    start_time = time.time()

    # Generate and write incrementally
    with open(temp_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['SMILES'])

        while valid_count < args.num_smiles:
            batch_smiles = generator.generate_batch(args.batch_size, args.temperature)
            total_generated += len(batch_smiles)

            for smi in batch_smiles:
                if valid_count >= args.num_smiles:
                    break
                if smi and is_valid_smiles(smi):
                    enc = encode_smiles(smi, predictor.char_to_idx, predictor.max_len)
                    if enc is not None:
                        writer.writerow([smi])
                        valid_count += 1

            if total_generated % (args.batch_size * 10) == 0:
                print(f"Generated {total_generated}, valid collected {valid_count}/{args.num_smiles}")

    gen_time = time.time() - start_time
    print(f"Generation finished. Total generated: {total_generated}, valid: {valid_count}")
    print(f"Generation time: {gen_time:.2f}s")

    # Predict in chunks from the temporary CSV
    print("Predicting properties...")
    output_csv = args.output
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pred_batch_size = 32
    first_chunk = True

    chunk_iter = pd.read_csv(temp_csv, chunksize=pred_batch_size)
    for chunk in chunk_iter:
        smiles_list = chunk['SMILES'].tolist()
        preds, valid_idx = predictor.predict_smiles_batch(smiles_list)
        if not preds:
            continue
        valid_smiles = [smiles_list[j] for j in valid_idx]
        df_chunk = pd.DataFrame({
            'SMILES': valid_smiles,
            'S1_exc': preds['S1_exc'],
            'T1_exc': preds['T1_exc'],
            'T2_exc': preds['T2_exc'],
        })
        # header only in the first chunk, then append
        df_chunk.to_csv(output_csv, mode='a', header=first_chunk, index=False)
        first_chunk = False

    # Clean up
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    print(f"Results saved to {output_csv}")
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
