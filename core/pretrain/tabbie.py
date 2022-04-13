import os
import sys
import json

import torch
import argparse
from pathlib import Path
from allennlp.common.util import import_module_and_submodules
from allennlp.commands import Subcommand

from core.pretrain.scripts.TabbieRunner import TabbieRunner
from core.pretrain.scripts.model_loadder import get_model_and_dataset_reader
from core.pretrain.scripts.util import load_yaml

sys.path += ['./scripts']


def cmd_builder(params, overrides):
    sys.argv = [
        "allennlp",  # command name, not used by main
        "predict",
        params['model_path'],
        params['pred_path'],
        "--output-file", str(Path(params['out_dir']) / params['out_pred_name']),
        "--predictor", "predictor",
        "--cuda-device", 0,
        "--batch-size", str(params['batch_size']),
        "-o", overrides,
    ]


def setup_env_variables(params):
    os.environ["ALLENNLP_DEBUG"] = "TRUE"
    for name, val in params.items():
        print(name, val)
        if isinstance(name, list):
            continue
        os.environ[name] = val


def get_table_embedding(embedding_file_path, data_num, config_file, input_json, save_step):
    # initialize
    params = load_yaml(config_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, params['cuda_devices']))
    del params['cuda_devices']
    setup_env_variables(params)

    # predict
    overrides = json.dumps({'dataset_reader': {'type': 'preprocess'}, 'trainer': {'opt_level': 'O0'}})
    cmd_builder(params, overrides)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="Commands", metavar="")

    for subcommand_name in sorted(Subcommand.list_available()):
        subcommand_class = Subcommand.by_name(subcommand_name)
        subcommand = subcommand_class()
        subparser = subcommand.add_subparser(subparsers)

    args = parser.parse_args()

    # scan registers
    import_module_and_submodules("core.pretrain")
    dataset_reader, model = get_model_and_dataset_reader(args, overrides)

    tabbie = TabbieRunner(model, dataset_reader)

    if input_json and len(input_json) > 0:
        table_embeddings = tabbie.run(args.batch_size, None, input_json, data_num, embedding_file_path, save_step)
    else:
        table_embeddings = tabbie.run(args.batch_size, args.input_file, None, data_num, embedding_file_path, save_step)

    print(len(table_embeddings))
    if len(table_embeddings) > 0:
        torch.save(table_embeddings, embedding_file_path)
    # return re
