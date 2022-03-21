import json
import logging
import sys
from itertools import islice
from typing import Iterator, TypeVar, Iterable, List

import numpy as np
import torch
from allennlp.common import JsonDict
from allennlp.data import Instance, DatasetReader
from tqdm import tqdm

try:
    from apex import amp
except ImportError:
    amp = None

from core.pretrain.models.model_tabbie import TabbieModel
from core.pretrain.scripts.util import cached_path

CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"
_DEFAULT_WEIGHTS = "best.th"


def save_embedding(embeddings, save_path, rank):
    save_path = save_path + "_" + str(rank) + ".pt"
    torch.save(embeddings, save_path)
    logging.info("save embedding file to" + save_path)

class TabbieRunner:

    def __init__(self, model: TabbieModel, dataset_reader: DatasetReader):
        self.module = model
        self.data_reader = dataset_reader

    def run(self, batch_size: int, input_file: str, input_json: list, total, save_path, save_step) -> List[JsonDict]:
        embeddings = []
        pbar = tqdm(total=total)
        rank = 0
        if input_file:
            for batch_json in lazy_groups_of(_get_json_data(input_file), batch_size):
                result = self._predict_json(batch_json)
                embeddings.extend(result)
                pbar.update(batch_size)
                if len(embeddings) >= save_step:
                    save_embedding(embeddings, save_path, rank)
                    rank += 1
                    embeddings.clear()
        else:
            for batch_json in lazy_groups_of(input_json, batch_size):
                result = self._predict_json(batch_json)
                embeddings.extend(result)
                pbar.update(batch_size)
                if len(embeddings) >= save_step:
                    save_embedding(embeddings, save_path, rank)
                    rank += 1
                    embeddings.clear()
        if len(embeddings > 0):
            save_embedding(embeddings, save_path, rank)
        return embeddings

    def _predict_json(self, batch_data: List[JsonDict]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = self.predict_json(batch_data[0])
        else:
            results = self.predict_batch_json(batch_data)
        return results

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance(instances)

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        line = json_dict
        table = line['table_data']
        idx = line['old_id'] if 'old_id' in line else line['id']

        max_rows, max_cols = 30, 20
        if len(table[0]) > max_cols:
            table = np.array(table)[:, :max_cols].tolist()

        table = table[0:max_rows]
        table_np = np.array(table)
        blank_loc = np.argwhere((table_np == '') | (table_np == '-') | (table_np == 'n/a') | (table_np == '&nbsp;'))

        instance = self.data_reader.text_to_instance(
            id=idx,
            header=table[0],
            table=table[1:max_rows],
            blank_loc=blank_loc,
        )
        return instance

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self.module.forward_on_instances(instances)
        return outputs

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self.module.forward_on_instance(instance)
        return outputs

A = TypeVar("A")


def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]:
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


def _get_json_data(input_file) -> Iterator[JsonDict]:
    if input_file == "-":
        for line in sys.stdin:
            if not line.isspace():
                yield json.loads(line)
    else:
        input_file = cached_path(input_file)
        with open(input_file, "r") as file_input:
            for line in file_input:
                if not line.isspace():
                    yield json.loads(line)



