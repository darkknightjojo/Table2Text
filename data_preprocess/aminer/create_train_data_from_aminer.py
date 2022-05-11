import json
import time
import os
import pathlib

fields = ['name', 'normalized_name', 'org', 'position', 'n_pubs', 'n_citation', 'h_index', 'tags', 'orgs']
h_index_threshold = 20
save_step = 10000
train = []
json_train = []
authors = []
train_path = ''
prefix = '/aminer_authors_2'

global save_file
global json_save_file
global author_save_file


def read_data(path, file):
    with open(path + "/" + file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            json_s = json.loads(line)
            # 过滤'h_index' < 20的学者
            if 'h_index' in json_s and int(json_s['h_index']) < h_index_threshold:
                line = f.readline()
                continue
            # 获取学者ID
            author_id = json_s['id'] + '\n'
            authors.append(author_id)
            # 获取序列数据
            train_data = generate_sequence_data(json_s)
            train.append(train_data + '\n')
            # 获取json数据
            train_data_json = generate_json_data(json_s)
            json_train.append(train_data_json)
            # 每save_step条保存一次
            if len(train) == save_step:
                save_file.writelines(train)
                json_save_file.writelines(json_train)
                author_save_file.writelines(authors)
                print("保存数据")
                train.clear()
                json_train.clear()
                authors.clear()
            # 读取下一条
            line = f.readline()


def generate_sequence_data(json_s):
    train_data = ''
    for key in fields:
        if key in json_s:
            # 单独处理tags
            if key == 'tags':
                tags = json_s[key]
                value = get_slice_of_array(tags, 3, 'tags')
            # 单独处理orgs
            elif key == 'orgs':
                orgs = json_s[key]
                #     只保留最多四个关联机构
                value = get_slice_of_array(orgs, 4)
            else:
                value = str(json_s[key]).replace("|", ",")
        else:
            continue

        values = value.split()
        current_field = ''
        for i, v in enumerate(values):
            current_field += v + '|' + key + '|' + str(i + 1) + '|' + str(len(values) - i) + ' '
        train_data += current_field
    return train_data


def generate_json_data(json_s):
    json_data = {}
    headers = []
    cells = []

    for key in fields:
        if key in json_s:
            headers.append(key)
            if key == 'tags':
                tags = json_s['tags']
                value = get_slice_of_array(tags, 3, 'tags')
                cells.append(value)
            elif key == 'orgs':
                orgs = json_s[key]
                #     只保留最多四个关联机构
                value = get_slice_of_array(orgs, 4)
                cells.append(value)
            else:
                cells.append(json_s[key])
    json_data['id'] = json_s['id']
    json_data['table_data'] = [headers, cells]

    train_data = json.dumps(json_data)
    return train_data + '\n'


def get_slice_of_array(array, num, data_type=''):
    value = ''
    length = min(num, len(array))
    for item in array[:length]:
        if data_type == 'tags':
            value += item['t'] + ' , '
        else:
            value += item.replace(', ', ' , ') + ' . '
    value = value.rstrip(" ").rstrip(",")
    return value[:400]


if __name__ == '__main__':
    current_path = os.getcwd() + prefix
    # 不存在train目录则创建
    if not pathlib.Path(current_path + "/train").exists():
        os.mkdir(current_path + "/train")
        os.mkdir(current_path + "/train/json")
        os.mkdir(current_path + "/train/authorId")
    start = time.time()
    path = current_path

    save_file = open(path + "/train" + prefix + "_train.txt", 'a+', encoding='utf-8')
    json_save_file = open(path + "/train/json" + prefix + "_train_json.txt", 'a+', encoding='utf-8')
    author_save_file = open(path + "/train/authorId" + prefix + "_authorId.txt", 'a+', encoding='utf-8')

    for index, file in enumerate(os.listdir(current_path)):
        if os.path.isfile(current_path + "/" + file):
            print("正在处理文件：" + file)
            read_data(current_path, file)
    #     剩余的未保存数据
    if len(train) > 0:
        save_file.writelines(train)
        json_save_file.writelines(json_train)
        author_save_file.writelines(authors)

    print("处理完毕！耗时：%10.3f" % (time.time() - start))
