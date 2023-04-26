import numpy as np
import random
import torch
import argparse

def seed_everything(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='asp2vec')

    # Essential parameters
    parser.add_argument('--seed', type=int, default=0, help="Seed for reproducibility")
    parser.add_argument('--dataset_seed', type=int, default=0, help="Seed for choosing dataset")
    parser.add_argument('--embedder', nargs='?', default='asp2vec')
    parser.add_argument('--dataset', nargs='?', default='filmtrust')
    parser.add_argument('--lr', type=float, default=0.003, help="Learning rate")
    parser.add_argument('--dim', type=int, default=20, help="Dimension size for each embedding vector (Total size: dim * num_aspects).")
    parser.add_argument('--num_aspects', type=int, default=5, help="Number of aspects")

    parser.add_argument('--isInit', action='store_true', default=False, help="Warm-up")
    parser.add_argument('--isReg', action='store_true', default=False, help="Aspect regularization")
    parser.add_argument('--reg_coef', type=float, default=0.01, help="\lambda in Eq.15")
    parser.add_argument('--threshold', type=float, default=0.5, help="\epsilon in Eq.13")
    parser.add_argument('--remove_percent', type=float, default=0.5, help="The portion of test dataset (Default: 50%).")

    # Default parameters
    parser.add_argument('--batch_size', type=int, default=50000)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--iter_max', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--path_length', type=int, default=80)
    parser.add_argument('--num_neg', type=int, default=2)
    parser.add_argument('--num_walks_per_node', type=int, default=10)
    parser.add_argument('--pooling', nargs='?', default='mean', help="How to Readout in Eq.9")
    parser.add_argument('--eval_freq', type=int, default=1)

    # Parameters related to Gumbel-softmax trick
    parser.add_argument('--isSoftmax', action='store_true', default=False)
    if parser.parse_known_args()[0].isSoftmax:
        parser.add_argument('--isGumbelSoftmax', action='store_true', default=False)
        parser.add_argument('--isNormalSoftmax', action='store_true', default=False)
        if parser.parse_known_args()[0].isGumbelSoftmax:
            parser.add_argument('--tau_gumbel', type=float, default=0.5, help="temperature in Eq.7")
            parser.add_argument('--isHard', action='store_true', default=False)  # Straight-through gumbel softmax

    return parser.parse_known_args()

# not used
def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)

def main():
    args, unknown = parse_args()
    print(args)

    seed_everything(args.seed)

    if args.embedder == 'asp2vec':
        from models import asp2vec
        embedder = asp2vec(args)

    embedder.training()

if __name__ == '__main__':
    main()
