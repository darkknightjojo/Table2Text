import json
import re

from tqdm import tqdm

from core.utils.parse import ArgumentParser


def create_pretrain_data(src_data, output_data, task_type, total):
    index = 0
    sp = '\\' # python里面识别不了'|'字符，所以用'\'代替
    prefix = 'train_' if task_type == "train" else 'test_'
    pbar = tqdm(total=total)
    with open(src_data, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            one = {}
            table_data = []
            headers = []
            cells = []

            array_ = line.split(" ")
            last_header = ''
            content = ''
            for a in array_:
                words = a.split(sp)
                # 不相等说明要读取新的header了
                if last_header != words[1]:
                    headers.append(words[1])
                    last_header = words[1]
                    if content != '':
                        cells.append(content)
                    content = words[0]
                else:
                    # 同一个header的内容
                    content += ' '
                    content += words[0]
            cells.append(content)
            # 读取完一行数据
            table_data.append(headers)
            table_data.append(cells)
            one['id'] = prefix + str(index)
            one['table_data'] = table_data
            with open(output_data, 'a') as jf:
                json.dump(one, jf)
                jf.write("\n")
            line = f.readline()
            index += 1
            pbar.update(1)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Simply add an argument for preprocess, train, translate
    parser.add_argument("--src_data", default='data/wikibio/full/train_input.txt', type=str,
                        help="src_data file path")
    parser.add_argument("--output_data", default='data/pretrain/input.jsonl', type=str,
                        help="json file for pretrain model")
    parser.add_argument("--task_type", default="train", type=str, help="task_type")
    parser.add_argument("--data_num", default=1000, type=int, help="train_data_num")

    params = parser.parse_args()
    create_pretrain_data(params.src_data, params.output_data, params.task_type, params.data_num)
