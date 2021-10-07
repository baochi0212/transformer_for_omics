import torch
import torch.nn as nn
from neural_net import embedding_encoder, embedding_decoder
from Encoder import encoderr
from Decoder import decoderr
from sklearn.metrics import mean_squared_error
import argparse
from Dataset import RNA_Protein
from torch.utils.data import Dataset, DataLoader

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_atac', type=str, default='public/cite/cite_adt_processed_training.h5ad')
    parser.add_argument('--path_gex', type=str, default='public/cite/cite_gex_processed_training.h5ad')
    return parser.parse_args()

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, embedding):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_encode = embedding[0]
        self.embed_decode = embedding[1]
        self.downsample = nn.Linear(512,128)
        self.upsample = nn.Linear(128,134)
    def forward(self, X, Y):
        X = self.embed_encode(X)
        Y = self.embed_decode(Y)
        X = self.downsample(self.encoder(X, None))
        state = self.decoder.init_state(X, None)
        return self.upsample(self.decoder(Y, state)[0])

if __name__ == "__main__":
    X = torch.rand([32,1,13953])
    Y = torch.rand([32,1,134])
    embed_encode = embedding_encoder(1,1,3)
    embed_decode = embedding_decoder(1,1,3)
    encode = encoderr(512,512,512,512,[1,512],512,1024,8,10,0.5)
    decode = decoderr(128,128,128,128,[1,128],128,256,8,10,0.5)
    model = EncoderDecoder(encode, decode, [embed_encode, embed_decode])
    # x = torch.rand([32,1,512])
    # y = torch.rand([32,1,128])
    # x = nn.Linear(512,128)(encode(x, None))
    # state = decode.init_state(x, None)
    # out = decode(y, state)[0]args = parse_arg()
    args = parse_arg()
    path_gex = args.path_gex
    path_atac = args.path_atac
    dataset = RNA_Protein(path_gex, path_atac)
    data_iter = DataLoader(dataset, batch_size=32, shuffle=True)
    X, Y = iter(data_iter).next()
    print(model(X, Y)[0].shape)
    