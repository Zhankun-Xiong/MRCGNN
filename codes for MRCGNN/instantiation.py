from torch.optim import Adam
from layer import MRCGNN
import numpy as np
import os
import random
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)

def Create_model(args):

    model = MRCGNN( feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2, decoder1=args.decoder1, dropout=args.dropout,zhongzi=args.zhongzi)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
    return model, optimizer