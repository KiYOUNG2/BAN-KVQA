"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import sys
import glob
import torch
from torch.utils.data import DataLoader

from solution_vqa.data.dataset import KvqaFeatureDataset, Dictionary
from solution_vqa.model import base_model
from solution_vqa.utils import dictionary_dict, set_seed
import solution_vqa.utils as utils
from solution_vqa.train import train

from args import (
    HfArgumentParser,
    MrcDataArguments,
    MrcModelArguments,
    MrcTrainingArguments,
    MrcProjectArguments,
)

if __name__ == '__main__':

    parser = HfArgumentParser(
        [
            MrcDataArguments,
            MrcModelArguments,
            MrcTrainingArguments,
            MrcProjectArguments
        ]
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    
    data_args, model_args, training_args, project_args = args

    set_seed(training_args.seed)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_path}")

    # wandb setting
    os.environ["WANDB_PROJECT"] = project_args.wandb_project

    # Set Logger
    utils.create_dir(training_args.output_dir)
    logger = utils.Logger(os.path.join(training_args.output_dir, 'log.txt'))
    logger.write(data_args.__repr__())
    logger.write(model_args.__repr__())
    logger.write(training_args.__repr__())
    logger.write(project_args.__repr__())

    # Load dictionary file
    if 'bert' in model_args.architectures:
        dictionary = None
    else:
        dictionary_path = os.path.join(
                                data_args.dataset_path,
                                dictionary_dict[model_args.architectures]['dict']
                                )
        dictionary = Dictionary.load_from_file(dictionary_path)

    # Load Dataset
    train_dset = KvqaFeatureDataset(
                                    split='train',
                                    dictionary=dictionary,
                                    max_length=data_args.max_seq_length,
                                    dataroot=data_args.dataset_path,
                                    tokenizer=dictionary_dict[model_args.architectures]['tokenizer']
                                )
    val_dset = KvqaFeatureDataset(
                                    split='val',
                                    dictionary=dictionary,
                                    max_length=data_args.max_seq_length,
                                    dataroot=data_args.dataset_path,
                                    tokenizer=dictionary_dict[model_args.architectures]['tokenizer']
                                )

    # Built Model
    constructor = 'build_%s' % model_args.model_init
    model = getattr(base_model, constructor)(
                        model_args.num_classes,
                        model_args.v_dim,
                        model_args.num_hid,
                        None,
                        model_args.op,
                        model_args.gamma,
                        model_args.architectures,
                        model_args.on_do_q,
                        model_args.finetune_q
                    ).cuda()

    if 'bert' not in model_args.architectures:
        model.q_emb.w_emb.init_embedding(os.path.join(
                                data_args.dataset_path,
                                dictionary_dict[model_args.architectures]['embedding']
                                )
                                )

    optim = None
    epoch = 0

    # load snapshot
    list_of_files = glob.glob(os.path.join(training_args.output_dir, '*.pth'))
    latest_file = max(list_of_files, key=os.path.getctime) if len(list_of_files) >= 1 else False
    if latest_file:
        print('loading %s' % latest_file) # TODO checkpoint file path
        model_data = torch.load(latest_file)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1
    
    train_loader = DataLoader(train_dset, training_args.per_device_train_batch_size, shuffle=True, num_workers=data_args.preprocessing_num_workers, collate_fn=utils.trim_collate)
    eval_loader = DataLoader(val_dset, training_args.per_device_eval_batch_size, shuffle=False, num_workers=data_args.preprocessing_num_workers, collate_fn=utils.trim_collate)
    print(optim)
    val_score, bound, train_n_type, val_n_type, train_type_score, val_type_score = \
        train(
            model,
            train_loader,
            eval_loader,
            training_args.num_train_epochs,
            os.path.join(training_args.output_dir),
            training_args.learning_rate,
            optim,
            epoch,
            logger
            )

    logger.write('\nVal upper bound: {}'.format(bound))
    logger.write('\nVal score: {}'.format(val_score))