import torch
import anndata as ad
import numpy as np
from torch.utils.data import DataLoader, Dataset
from einops import repeat

class RNA_Protein(Dataset):
    def __init__(self, gex_path, atac_path):
        self.file_gex = ad.read_h5ad(gex_path).X[:1000,:]
        self.file_atac = ad.read_h5ad(atac_path).X[:1000,:]
    
    def __getitem__(self, index):
        X = torch.tensor(self.file_gex[index,:].todense()).float()
        Y = torch.tensor(self.file_atac[index,:].todense()).float()
        return X,Y
    def __len__(self):
        return self.file_gex.shape[0]

if __name__ == '__main__':
    atac_path = 'public/multiome/multiome_atac_processed_training.h5ad'
    gex_path = 'public/multiome/multiome_atac_processed_training.h5ad'
    data = RNA_Protein(gex_path, atac_path)
    print(data[10][0].shape)



