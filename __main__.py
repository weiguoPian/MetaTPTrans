import argparse
from ast import arg
from statistics import mode

from torch.utils.data import DataLoader
from dataset import PathAttenDataset, TextVocab, UniTextVocab, collect_fn, CTTextVocab
from trainer import metaTrainer
from model import metaModel
import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default='summarization', choices=['completion', 'summarization'])
    parser.add_argument("--MultiStepLR", type=boolean_string, default=False, help="")
    parser.add_argument("--milestones", type=int, default=[25, 30], nargs='+', help="")
    parser.add_argument("--model_type", type=str, default='alpha', choices=['alpha', 'beta', 'gamma'], help="")

    parser.add_argument("--on_memory", type=boolean_string, default=True, help="Loading datasets into memory")

    # dataset size
    parser.add_argument("--max_code_length", type=int, default=512, help="")
    parser.add_argument("--max_path_length", type=int, default=32, help="")
    parser.add_argument("--max_r_path_length", type=int, default=32, help="")
    parser.add_argument("--max_path_num", type=int, default=512, help="the num of unique relative path")
    parser.add_argument("--max_r_path_num", type=int, default=256, help="the num of unique absolute path")
    parser.add_argument("--max_target_len", type=int, default=7, help=' <eos or sos> + true len of method name')

    # vocab
    parser.add_argument("--s_vocab_portion", type=float, default=0.999, help="not work for ct_vocab")
    parser.add_argument("--t_vocab_portion", type=float, default=1, help="not work for ct_vocab")
    parser.add_argument("--vocab_threshold", type=int, default=100, help="not work for ct_vocab")
    parser.add_argument("--uni_vocab", type=boolean_string, default=True,
                        help="source vocab (embedding) = target vocab (embedding)")
    parser.add_argument("--weight_tying", type=boolean_string, default=True,
                        help="right embedding = pre softmax matrix ")
    parser.add_argument("--ct_vocab", type=boolean_string, default=False, help="use code transformer's voc")

    # trainer
    parser.add_argument("--with_cuda", type=boolean_string, default=True, help="training with CUDA: true or false")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--lr_scheduler", type=boolean_string, default=True,
                        help="We use the ReduceLROnPlateau scheduler")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="minimal learning rate of adam")
    parser.add_argument("--patience", type=int, default=0, help="patience")
    parser.add_argument("--clip", type=float, default=0, help="0 is no clip")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--accu_batch_size", type=int, default=128,
                        help="number of real batch_size per step, save gpu memory")
    parser.add_argument("--val_batch_size", type=int, default=128, help="number of batch_size of valid")
    parser.add_argument("--infer_batch_size", type=int, default=128, help="number of batch_size of infer")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=28, help="dataloader worker size")
    parser.add_argument("--save", type=boolean_string, default=True, help="whether to save model checkpoint")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
    parser.add_argument("--dropout", type=float, default=0.2, help="")
    parser.add_argument("--shuffle", type=boolean_string, default=True, help="whether to shuffle the training data")

    # glove
    parser.add_argument("--pretrain", type=boolean_string, default=False,
                        help="Whether to use the glove pretrain embedding")
    parser.add_argument("--embedding_file", type=str, default='', help="file path to glove txt")

    # path embedding
    parser.add_argument("--path_embedding_size", type=int, default=64, help="embedding size of path node")
    parser.add_argument("--path_embedding_num", type=int, default=335,
                        help="total node type num, and also be used as padding idx in path."
                             "You can also set it for different language: Python: 109; Ruby: 105; Javascript: 105; Go:94;"
                             "And you can also choose a number such as 120 bigger than all of them")
    parser.add_argument("--bidirectional", type=boolean_string, default=True, help="for path gru")
    parser.add_argument("--gru_size", type=int, default=64, help="for path gru")
    parser.add_argument("--gru_layers", type=int, default=1, help="for path gru")
    parser.add_argument("--meta_output", type=boolean_string, default=True, help="")

    # transformer
    parser.add_argument("--embedding_size", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("--activation", type=str, default='gelu', help="", choices=['gelu', 'relu'])
    parser.add_argument("--hidden", type=int, default=1024, help="hidden size of transformer model")
    parser.add_argument("--d_ff_fold", type=int, default=4, help="ff_hidden = ff_fold * hidden; for decoder")
    parser.add_argument("--e_ff_fold", type=int, default=4, help="ff_hidden = ff_fold * hidden; for encoder")
    parser.add_argument("--layers", type=int, default=3, help="number of encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=3, help="number of decoder layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")

    # Path encoding
    parser.add_argument("--relation_path", type=boolean_string, default=True, help="Whether to use relative path")
    parser.add_argument("--absolute_path", type=boolean_string, default=True, help="Whether to use absolute path")
    parser.add_argument("--path_value", type=boolean_string, default=True,
                        help="Whether to use the weight sum of Value of relative path")
    parser.add_argument("--ap_kq", type=boolean_string, default=True,
                        help="The projection of Key and Query for absolute path encoding")
    parser.add_argument("--rp_kv", type=boolean_string, default=True,
                        help="The projection of Key and Value for relative path encoding")

    # Ablation study for norm and hops
    parser.add_argument("--gru_ln", type=boolean_string, default=True, help="The normalization for gru")
    parser.add_argument("--hop", type=boolean_string, default=False, help="convert path information to hops")

    # Other model triggers
    parser.add_argument("--absolute_position", type=boolean_string, default=True,
                        help="The vanilla absolute positional encoding for transformer")
    parser.add_argument("--embedding_mul", type=boolean_string, default=True, help="For word embedding")
    parser.add_argument("--pointer", type=boolean_string, default=True, help="whether to use pointer network")
    parser.add_argument("--pointer_res", type=boolean_string, default=False,
                        help="whether to use res connection for pointer")
    parser.add_argument("--pointer_type", type=str, choices=['mul', 'add'], default='mul', help="")

    # Some not useful designs, can also ignore them
    parser.add_argument("--sqrt_norm", type=int, default=1,
                        help="set the sqrt(2d) like TUPE")
    parser.add_argument("--ap_split", type=boolean_string, default=False,
                        help="set two different embedding matrix for absolute and relative path encoding")
    parser.add_argument("--is_named", type=boolean_string, default=True,
                        help="A trigger for whether to use parser named encoding")

    # Other Training setting
    parser.add_argument("--unk_shift", type=boolean_string, default=True,
                        help="reduce the prob of unk to avoid inferring unk token")
    parser.add_argument("--seed", type=boolean_string, default=True, help="fix seed or not")
    parser.add_argument("--seed_idx", type=int, default=20, help="choose different seed idx for error bars")
    parser.add_argument("--old_calculate", type=boolean_string, default=True,
                        help="see the statistic.py in trainer dir for details")

    # Debug setting
    parser.add_argument("--tiny_data", type=int, default=0, help="pick some tiny data for debug")
    parser.add_argument("--data_debug", type=boolean_string, default=False, help="try to over-fit on valid data")
    parser.add_argument("--train", type=boolean_string, default=True, help="Whether to train")
    parser.add_argument("--test", type=boolean_string, default=True, help="Whether to test")
    parser.add_argument("--load_checkpoint", type=boolean_string, default=False,
                        help="load checkpoint for continue train or infer")
    parser.add_argument("--checkpoint", type=str, default='', help="the checkpoint file path")

    parser.add_argument("--lan_embedding_dim", type=int, default=1024, help="")
    parser.add_argument("--projection_dim", type=int, default=2048, help="")

    args = parser.parse_args()
    if args.seed:
        setup_seed(args.seed_idx)
    if args.ct_vocab:
        s_vocab = CTTextVocab(args)
        t_vocab = s_vocab
    else:
        if args.uni_vocab:
            s_vocab = UniTextVocab(args)
            t_vocab = s_vocab
        else:
            s_vocab = TextVocab(args, 'source')
            t_vocab = TextVocab(args, 'target')

    print("Loading Train Dataset")
    if args.data_debug:
        train_dataset = PathAttenDataset(args, s_vocab, t_vocab, type_='valid')
    else:
        train_dataset = PathAttenDataset(args, s_vocab, t_vocab, type_='train')

    print("Loading Valid Dataset")
    valid_dataset = PathAttenDataset(args, s_vocab, t_vocab, type_='valid')
    print("Loading Test Dataset")
    test_dataset = PathAttenDataset(args, s_vocab, t_vocab, type_='test')
    if args.on_memory:
        num_workers = args.num_workers
    else:
        num_workers = 0

    print("Creating Dataloader")
    if args.train:
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers,
                                       shuffle=args.shuffle, collate_fn=collect_fn)
    else:
        train_data_loader = None
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, num_workers=num_workers,
                                   collate_fn=collect_fn)
    valid_infer_data_loader = DataLoader(valid_dataset, batch_size=args.infer_batch_size, num_workers=num_workers,
                                         collate_fn=collect_fn)
    test_infer_data_loader = DataLoader(test_dataset, batch_size=args.infer_batch_size, num_workers=num_workers,
                                        collate_fn=collect_fn)
    print("Building Model")
    model = metaModel(args, s_vocab, t_vocab)

    print("Creating Trainer")
    trainer = metaTrainer(args=args, model=model, train_data=train_data_loader, valid_data=valid_data_loader,
                              valid_infer_data=valid_infer_data_loader, test_infer_data=test_infer_data_loader, t_vocab=t_vocab)
    
    if args.load_checkpoint:
        checkpoint_path = 'checkpoint/{}'.format(args.checkpoint)
        trainer.load(checkpoint_path)
    print("Training Start")

    for epoch in range(args.epochs):
        if args.train:
            trainer.train(epoch)
        trainer.predict_multi(epoch, test=False)
        trainer.predict_multi(epoch, test=True)
    trainer.writer.close()



if __name__ == '__main__':
    train()
