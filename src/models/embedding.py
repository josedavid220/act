import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(
        self,
        n_feats,
        token_size,
    ):
        super(Embedding, self).__init__()
        self.token_size = token_size
        embedding_dim = n_feats * token_size
        flatten_dim = embedding_dim

        self.linear_encoding = nn.Linear(flatten_dim, embedding_dim)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = F.unfold(x, kernel_size=(1, self.token_size), stride=(1, self.token_size))
        x = x.transpose(1, 2)
        
        return self.linear_encoding(x) + x # Get the embedding and add the residual connection
    
def unembedding(x, token_size, seq_len):
    x = x.transpose(2, 1)
    x = F.fold(
        x,
        output_size=(1, seq_len),
        kernel_size=(1, token_size),
        stride=token_size,
    )
    return x.squeeze()