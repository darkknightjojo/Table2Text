import json

from tqdm import tqdm

from core.utils.parse import ArgumentParser
from contextlib import ExitStack


def create_train_data(src_path, target_path, task_type, data_num):
    pbar = tqdm(total=data_num)
    with open(src_path, 'r', encoding='utf-8') as input_json:
        index = 0
        line = input_json.readline()
        while index < data_num and line:
            json_data = json.loads(line)
            val = json_data["table_data"][1]
            content = ""
            for v in val:
                words = v.split(" ")
                cell = ''
                for word in words:
                    cell += word
                    cell += '-'
                content += cell.strip('-')
                content += ' '

            with open(target_path, 'a+', encoding='utf-8') as f:
                f.write(content + '\n')
            line = input_json.readline()
            pbar.update(1)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Simply add an argument for preprocess, train, translate
    parser.add_argument("--src_data", default='data/pretrain/input.json', type=str,
                        help="src_data file path")
    parser.add_argument("--output_data", default='data/wikibio/full/train_data.txt', type=str,
                        help="json file for pretrain model")
    parser.add_argument("--task_type", default="train", type=str, help="task_type")
    parser.add_argument("--data_num", default=1000, type=int, help="train_data_num")

    params = parser.parse_args()

    create_train_data(params.src_data, params.output_data, params.task_type, params.data_num)