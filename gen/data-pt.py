import torch
import random

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# Load SMILES data
with open("data.txt", "r") as f:
    smiles_list = [line.strip() for line in f.readlines()]

# Build vocabulary
special_tokens = ["?", "+"]  # start and padding symbols
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

# Vocabulary: atom symbols first, then special tokens
vocab = sorted(list(atom_symbols)) + special_tokens
vocab_dict = {char: idx for idx, char in enumerate(vocab)}
reverse_vocab_dict = {idx: char for char, idx in vocab_dict.items()}

# Maximum sequence length (plus the start symbol "?")
max_len = max(len(smiles) for smiles in smiles_list) + 1

vocab_complete = {
    "char_to_idx": vocab_dict,
    "idx_to_char": reverse_vocab_dict,
    "max_len": max_len
}
torch.save(vocab_complete, "vocab.pt")

def encode_smiles(smiles, vocab_dict, max_len):
    """Encode a SMILES string, padded to max_len with '+'."""
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
        encoded.append(vocab_dict["+"])  # padding
    return encoded

# Encode all SMILES
encoded_smiles = [encode_smiles(smiles, vocab_dict, max_len) for smiles in smiles_list]

smiles_tensor = torch.tensor(encoded_smiles, dtype=torch.long)

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

train_tensor = smiles_tensor[train_indices]
val_tensor = smiles_tensor[val_indices]
test_tensor = smiles_tensor[test_indices]

# Save the split dataset
torch.save({
    "train": train_tensor,
    "val": val_tensor,
    "test": test_tensor
}, "smiles.pt")

# Summary
print(f"Max length (saved to vocab.pt): {max_len}")
print("Vocabulary:")
print(vocab_complete)
print("\nTrain size:", train_tensor.size(0))
print("Val size:", val_tensor.size(0))
print("Test size:", test_tensor.size(0))
print("\nEncoded representation of the last molecule (test set):")
print(test_tensor[-1].tolist())
