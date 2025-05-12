import torch
# Example tensors
batch_size = 3
seq_len = 4
vocab_size = 5
probs = torch.rand(batch_size, seq_len, vocab_size)  # Example probability tensor
indices = torch.randint(0, vocab_size, (batch_size, seq_len))  # Example indices tensor
# Use gather to extract the probabilities of the tokens in indices
# Add an extra dimension to indices to match the dimensionality required by gather
indices = indices.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
# Gather the probabilities
extracted_probs = torch.gather(probs, 2, indices)  # Shape: [batch_size, seq_len, 1]
# If you want the result to be in shape [batch_size, seq_len], squeeze the last dimension
extracted_probs = extracted_probs.squeeze(-1)  # Shape: [batch_size, seq_len]
print(probs)
print(indices)
print(extracted_probs)