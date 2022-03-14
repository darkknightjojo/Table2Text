import torch

from core.pretrain.tabbie import get_table_embedding

if __name__ == '__main__':
    get_table_embedding(None)
    # x = torch.load("../../experiments/wikibio/pretrain/table_embedding.pt")
    # print(x[0])