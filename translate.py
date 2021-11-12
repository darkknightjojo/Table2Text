#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from core.utils.logging import init_logger
from core.utils.misc import split_corpus
from core.translate.translator import build_translator

import core.opts as opts
from core.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )
        
    return opt


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main(args=None):
    parser = _get_parser()

    opt = parser.parse_args(args) if args else parser.parse_args()
    opt = translate(opt)
    
    return opt


if __name__ == "__main__":
    _ = main()
