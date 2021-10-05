import torch
import torch.nn as nn
#this embedding only can be applied in words modelling 
class Embedding_(nn.Module):
    def __init__(self, dim_matrix, embed_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(dim_matrix, embed_dim)
    def forward(self, x):
        return self.embedding(x)

if __name__ == '__main__':
    embed = Embedding_(1000)
    x = torch.ones([3,100], dtype=torch.long)
    print(embed(x).shape)


