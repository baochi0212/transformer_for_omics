import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from Dataset import RNA_Protein
from sklearn.model_selection import train_test_split
from model import EncoderDecoder
from Encoder import encoderr
from Decoder import decoderr
from tqdm import tqdm 
from neural_net import embedding_encoder, embedding_decoder


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_atac', type=str, default='public/cite/cite_adt_processed_training.h5ad')
    parser.add_argument('--path_gex', type=str, default='public/cite/cite_gex_processed_training.h5ad')
    return parser.parse_args()

def train_seq2seq(data_iter, model, criterion, num_epochs, optimizer):
    model.train()
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0 
        for src, trg in tqdm(data_iter):
            optimizer.zero_grad()
            pred = model(src, trg)
            loss = criterion(pred, trg)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}: The loss is {total_loss}')
    

if __name__ == '__main__':
    args = parse_arg()
    path_gex = args.path_gex
    path_atac = args.path_atac
    dataset = RNA_Protein(path_gex, path_atac)
    data_iter = DataLoader(dataset, batch_size=32, shuffle=True)
    encoder = encoderr(512,512,512,512,[1,512],512,1024,8,10,0.5)
    decoder = decoderr(128,128,128,128,[1,128],128,256,8,10,0.5)
    model = EncoderDecoder(encoder, decoder, [embedding_encoder(1,1,3), embedding_decoder(1,1,3)])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    num_epochs = 10
    train_seq2seq(data_iter, model, criterion, num_epochs, optimizer)
 


