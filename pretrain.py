import torch

from core.pretrain.tabbie import get_table_embedding
from core.utils.parse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()

    # Simply add an argument for preprocess, train, translate
    parser.add_argument("--embedding_file_path", default='experiments/wikbio/pretrain/embeddings/table_embedding', type=str,
                        help="embedding file path")
    parser.add_argument("--data_num", default=10000, type=int,
                        help="data num")
    parser.add_argument("--config_file", default='core/pretrain/cfg/tabbie_ft_col.yml', type=str,
                        help="embedding file path")
    parser.add_argument("--save_step", default=20000, type=int,
                        help="save_step, 参考batch_size设置，否则读取时会跨文件")

    params = parser.parse_args()
    get_table_embedding(params.embedding_file_path, params.data_num, params.config_file, None, params.save_step)
    # x = torch.load("../../experiments/wikibio/pretrain/table_embedding.pt")
    # print(x[0])