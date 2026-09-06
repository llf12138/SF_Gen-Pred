import torch
import random
import pandas as pd

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# Load CSV data
df = pd.read_csv("FORMED.csv")
smiles_list = df['smiles'].tolist()

# Target columns: edit this line to add or remove targets
target_columns = ['S1_exc', 'T1_exc', 'T2_exc']

target_dict = {}
for col in target_columns:
    target_dict[col] = df[col].tolist()

# Build vocabulary
atom_symbols = set()

# Parse SMILES and collect tokens
for smiles in smiles_list:
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":  # bracketed tokens such as [Ge]
            j = smiles.find("]", i)
            if j != -1:
                atom_symbols.add(smiles[i : j + 1])
                i = j + 1
            else:
                raise ValueError(f"Unmatched bracket in: {smiles}")
        elif i + 1 < len(smiles) and smiles[i : i + 2] in ["Cl", "Br"]:  # two-letter symbols
            atom_symbols.add(smiles[i : i + 2])
            i += 2
        else:
            atom_symbols.add(smiles[i])  # single-character symbols
            i += 1

# Vocabulary: padding symbol "+" is forced to index 0, "?" is the start symbol
vocab = ["+"] + sorted(list(atom_symbols)) + ["?"]
vocab_dict = {char: idx for idx, char in enumerate(vocab)}
reverse_vocab_dict = {idx: char for char, idx in vocab_dict.items()}

# Maximum sequence length (plus the start symbol)
max_len = max(len(smiles) for smiles in smiles_list) + 1

vocab_complete = {
    "char_to_idx": vocab_dict,
    "idx_to_char": reverse_vocab_dict,
    "max_len": max_len
}
torch.save(vocab_complete, "vocab.pt")

def encode_smiles(smiles, vocab_dict, max_len):
    """Encode a SMILES string, padded to max_len with index 0 ("+")."""
    encoded = [vocab_dict["?"]]  # start symbol
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":  # bracketed tokens
            j = smiles.find("]", i)
            if j != -1:
                token = smiles[i : j + 1]
                if token not in vocab_dict:
                    raise ValueError(f"Unknown token: {token}")
                encoded.append(vocab_dict[token])
                i = j + 1
            else:
                raise ValueError(f"Unmatched bracket in: {smiles}")
        elif i + 1 < len(smiles) and smiles[i : i + 2] in vocab_dict:  # two-letter symbols
            token = smiles[i : i + 2]
            encoded.append(vocab_dict[token])
            i += 2
        else:
            token = smiles[i]
            if token not in vocab_dict:
                raise ValueError(f"Unknown token: {token}")
            encoded.append(vocab_dict[token])
            i += 1
    while len(encoded) < max_len:
        encoded.append(0)  # padding with index 0 ("+")
    return encoded

# Encode all SMILES
encoded_smiles = [encode_smiles(smiles, vocab_dict, max_len) for smiles in smiles_list]

smiles_tensor = torch.tensor(encoded_smiles, dtype=torch.long)

target_tensors = {}
for col in target_columns:
    target_tensors[col] = torch.tensor(target_dict[col], dtype=torch.float)

# Train/val/test split (80/10/10)
total_samples = len(smiles_tensor)
indices = list(range(total_samples))
random.shuffle(indices)

train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size : train_size + val_size]
test_indices = indices[train_size + val_size :]

train_smiles = smiles_tensor[train_indices]
val_smiles = smiles_tensor[val_indices]
test_smiles = smiles_tensor[test_indices]

train_targets = {}
val_targets = {}
test_targets = {}

for col in target_columns:
    train_targets[col] = target_tensors[col][train_indices]
    val_targets[col] = target_tensors[col][val_indices]
    test_targets[col] = target_tensors[col][test_indices]

# Save the split dataset (SMILES + multiple targets)
torch.save({
    "train_smiles": train_smiles,
    "train_targets": train_targets,
    "val_smiles": val_smiles,
    "val_targets": val_targets,
    "test_smiles": test_smiles,
    "test_targets": test_targets
}, "smiles.pt")

# Summary
print(f"Max sequence length: {max_len}")
print("Vocabulary:")
print(f"Vocab size: {len(vocab)}")
print(f"Padding symbol '+' index: {vocab_dict['+']}")
print(f"Start symbol '?' index: {vocab_dict['?']}")
print(f"\nTrain size: {train_smiles.size(0)}")
print(f"Val size: {val_smiles.size(0)}")
print(f"Test size: {test_smiles.size(0)}")

print(f"\nTarget statistics:")
for col in target_columns:
    tensor = target_tensors[col]
    print(f"{col} range: {tensor.min().item():.3f} ~ {tensor.max().item():.3f}")
    print(f"{col} mean: {tensor.mean().item():.3f} +/- {tensor.std().item():.3f}")

print("\nEncoded representation of the last sample (test set):")
print(f"SMILES encoding: {test_smiles[-1].tolist()}")
for col in target_columns:
    print(f"{col} value: {test_targets[col][-1].item():.3f}")

# Verify vocab.pt contents
loaded_vocab = torch.load("vocab.pt")
print(f"\nvocab.pt verification:")
print(f"Keys: {list(loaded_vocab.keys())}")
print(f"max_len: {loaded_vocab['max_len']}")
