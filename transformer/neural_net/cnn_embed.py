import torch
import torch.nn as nn

#EMBEDDING WITH CONV-1D
class embedding_encoder(nn.Module):
    def __init__ (self, dim_in, dim_out, dim_hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(dim_in, dim_hidden, kernel_size=5, padding=1, stride=3)
        self.conv2 = nn.Conv1d(dim_hidden, dim_hidden, kernel_size=3, padding=1, stride=3)
        self.conv3 = nn.Conv1d(dim_hidden, dim_out, kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU()
        self.Linear = nn.Linear(776, 512)
    def forward(self, X):
        return self.Linear(self.relu(self.conv3(self.conv2(self.conv1(X)))))

class embedding_decoder(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(dim_in, dim_hidden, kernel_size=5, padding=1, stride=1)
        self.conv2 = nn.Conv1d(dim_hidden, dim_hidden, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(dim_hidden, dim_out, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
    def forward(self, X):
        return self.conv3(self.conv2(self.conv1(X)))


if __name__ == "__main__":
    x = torch.rand([32,1,134])
    y = torch.rand([32,1,13953])
    print(embedding_encoder(1,1,3)(y).shape)
    print(embedding_decoder(1,1,3)(x).shape)
