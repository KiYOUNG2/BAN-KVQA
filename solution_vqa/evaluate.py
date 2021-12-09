"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import KvqaFeatureDataset, Dictionary
import model.base_model as base_model
from solution_vqa.train import evaluate
from .utils import utils
from .utils.registry import dictionary_dict
from .utils.constants import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--q_emb', type=str, default='glove-rg', choices=dictionary_dict.keys())
    parser.add_argument('--op', type=str, default='')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--finetune_q', action='store_true', help='finetune question embedding?')
    parser.add_argument('--on_do_q', action='store_true', help='turn on dropout of question embedding?')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban-kvqa')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'eval_log.txt'))
    logger.write(args.__repr__())

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.input is None:
        args.input = args.output


    if 'bert' in args.q_emb:
        dictionary = None
    else:
        dictionary_path = os.path.join(args.dataroot, dictionary_dict[args.q_emb]['dict'])
        dictionary = Dictionary.load_from_file(dictionary_path)
    val_dset = KvqaFeatureDataset('val', dictionary, tokenizer=dictionary_dict[args.q_emb]['tokenizer'])

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(val_dset, args.num_hid, args.op, args.gamma,
                                             args.q_emb, args.on_do_q, args.finetune_q).cuda()

    model = nn.DataParallel(model).cuda()

    optim = None
    epoch = 0

    # load snapshot
    if args.input is not None:
        path = os.path.join(args.output)
        print('loading %s' % path)

        model_data = torch.load(glob.glob(os.path.join(path, "model_epoch*.pth"))[-1])
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1

    eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    model.train(False)
    val_score, zcore, bound, entropy, val_n_type, val_type_score = evaluate(model, eval_loader)

    logger.write('\nMean val upper bound: {}'.format(bound))
    logger.write('\nMean val score: {}'.format(val_score))
    logger.write('\nAnswer type: '+', '.join(val_dset.idx2type))
    logger.write('\n'+'Number of examples for each type on val: {}'.format(val_n_type))
    logger.write('\n'+'Mean score for each type on val: {}'.format(val_type_score / val_n_type))
