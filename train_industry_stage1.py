import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
# from utils import *
from stage1_trainer import model_train
from model.stage1_MF import MF
import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
import warnings
import math
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='industry', help='Dataset name:  kuairand, industry,amazon_book, ml-1m')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=999, help='Random seed') 
parser.add_argument('--is_user', type=bool, default=True, help='with user_emb.')  
# parser.add_argument('--max_len', type=int, default=200, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch Size')  
parser.add_argument('--test_batch_size', type=int, default=1024
                    , help='test batch size')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument("--hidden_dims", default=32, type=int, help="hidden dims of model")
parser.add_argument('--dropout', type=float, default=0, help='Dropout of representation')
# parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
# parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument('--data_latent_dim', type=int, default=128, 
                    help='user/item latent embedding size')
# parser.add_argument('--transformer_enc_dim', type=int, default=32, 
#                     help='item encoding size')
parser.add_argument('--transformer_n_layer', type=int, default=2, 
                            help='number of encoder layers in transformer')
parser.add_argument('--transformer_n_head', type=int, default=4, 
                            help='number of attention heads in transformer')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')  ## 500
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5,10,20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd'], help='Optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step')
parser.add_argument('--stage1_choose', type=int, default=50, 
                    help='user/item latent embedding size')
parser.add_argument('--slate_offset', type=float, default=1.1)
# parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=10, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--cuda', type=int, default=7, help='cuda device number; set to -1 (default) if using cpu')

args = parser.parse_args()

print(args)


if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)



def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class CustomDataset_train(Dataset):
    def __init__(self, dataframe,args):
        self.dataframe = dataframe
        self.dataset=args.dataset
        self.item_num=args.item_num

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user = self.dataframe.iloc[idx]['user']
        # item= torch.tensor(self.dataframe.iloc[idx]['item'], dtype=torch.float32)

        pos= torch.tensor(random.choice(self.dataframe.iloc[idx]['item']), dtype=torch.float32)
       
        # neg = torch.randint(0, self.item_num, (self.negative_num,))
        # item= torch.tensor(self.dataframe.iloc[idx]['item'], dtype=torch.float32)
       
        # label = torch.tensor(self.dataframe.iloc[idx]['y'], dtype=torch.float32)

        return user, pos

class CustomDataset_test(Dataset):
    def __init__(self, dataframe,args):
        self.dataframe = dataframe
        self.dataset=args.dataset

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user = self.dataframe.iloc[idx]['user']
        item= torch.tensor(self.dataframe.iloc[idx]['items'], dtype=torch.float32)
       
        label = torch.tensor(self.dataframe.iloc[idx]['y'], dtype=torch.float32)
       
        return user, item,label


def dataset_create(args, statis):
    args.item_num = statis['item_num'][0]
    args.max_len = statis['his_len'][0]
    args.user_num = statis['user_num'][0]
    return args


def main(args):    
    fix_random_seed_as(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    statis_path = 'dataset/'+args.dataset+'/data_statis.df'
    statis=pd.read_pickle(statis_path)
    item_score_path = 'dataset/'+args.dataset+'/popularity_score.pt'
    item_score=torch.load(item_score_path).to(device)

    args = dataset_create(args, statis)
    train_data_path = 'dataset/'+args.dataset+ "/train_stage1_new.df"
    train_df=pd.read_pickle(train_data_path)
    train_dataset = CustomDataset_train(train_df,args)

    test_data_path = 'dataset/'+args.dataset+ "/test_stage1.df"
    test_df=pd.read_pickle(test_data_path)
    test_dataset = CustomDataset_test(test_df,args)
    import pdb
    pdb.set_trace()


    
   
    
    
    logger.info(args)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    # diffu_rec = create_model_diffu(args)
    model = MF(args)
    
    # best_model, test_results = model_train(train_data_loader, val_data_loader, test_data_loader, model, args, logger)

    # for index_temp, train_batch in enumerate(train_data_loader):
    #         train_batch = [x.to(device) for x in train_batch]
            
    loss= model_train(train_data_loader, test_data_loader, model, args, logger,item_score,device)



if __name__ == '__main__':
    main(args)
