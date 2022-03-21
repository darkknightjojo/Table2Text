import os

import torch

if __name__=='__main__':
    path = "experiments/wikibio/pretrain/embeddings/train_table_embedding_20w_"
    dir_path = "/"
    paths = path.split("/")[:-1]
    dir_path = dir_path.join(paths) + "/"
    file_names = os.listdir(dir_path)
    for file in file_names:
        embeddings = torch.load(dir_path + file)
        prefix = file.split("_")
        rank = int(prefix[-1].replace('.pt', ''))
        prefix = prefix[:-1]
        length = len(embeddings)
        new_embeddings = embeddings[:length//2]
        embeddings = embeddings[length//2:]
        new_file_1 = "_".join(prefix[:-1]) + '_' + str(2 * rank) + '.pt'
        new_file_2 = "_".join(prefix[:-1]) + '_' + str(2 * rank + 1) + '.pt'
        torch.save(new_embeddings, dir_path + new_file_1)
        torch.save(embeddings, dir_path + new_file_2)