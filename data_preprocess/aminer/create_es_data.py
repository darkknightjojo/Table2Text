# 生成ES格式（bulk_api）的数据

import json
import os

current_path = os.getcwd() + '/aminer_authors_1'


def deal_data(file_name, index):
    data = []
    new_jsonfile = open(current_path + '/es/aminer_authors_1_' + str(index) + '.json', 'w', encoding='UTF-8')
    with open(file_name, 'r', encoding='utf-8') as fp:
        line = fp.readline()
        while line:
            try:
                j = json.loads(line)
                new_data = {}
                new_data['index'] = {}
                new_data['index']['_index'] = "aminer_authors"
                new_data['index']['_id'] = str(j['id'])
                temp = json.dumps(new_data).encode("utf-8").decode('unicode_escape')
                data.append(temp + '\n')
                data.append(line)
                # 每一万写入一次
                if len(data) == 10000:
                    new_jsonfile.writelines(data)
                    data.clear()
                line = fp.readline()
            except Exception:
                pass
    if len(data) > 0:
        new_jsonfile.writelines(data)
    new_jsonfile.close()


if __name__ == '__main__':
    # index = 1
    for index, file in enumerate(os.listdir(current_path)):
        print(file)
        if os.path.isfile(file):
            deal_data(file, index)