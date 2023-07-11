import os

from parse import parse_args
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import time

import numpy as np
import copy
import pickle

from utils import *
from evaluation import *
from model import *
from dataprocess import *

if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)

    """build dataset"""
    train_cf, val_cf, test_cf, user_dict, args.n_users, args.n_items, clicked_set, adj = load_data(args)

    print(args.n_users, args.n_items, train_cf.shape[0] + test_cf.shape[0],
        (train_cf.shape[0] + test_cf.shape[0]) / (args.n_items * args.n_users))

    if(args.neg_in_val_test == 1):  # whether negative samples from validation and test sets
        clicked_set = user_dict['train_user_set']

    """build model"""
    if args.model == 'LightGCN':
        model = LightGCN(args).to(args.device)
    elif args.model == 'NGCF':
        model = NGCF(args).to(args.device)
    elif args.model == 'MF':
        model = MF(args).to(args.device)
    elif args.model == 'CAGCN' or args.model == "CAGCN-fusion":
        model = CAGCN(args).to(args.device)

    adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
    edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

    model.adj_sp_norm = adj_sp_norm.to(args.device)
    model.edge_index = edge_index.to(args.device)
    model.edge_weight = edge_weight.to(args.device)
    model.deg = deg.to(args.device)

    # trend for model CAGCN
    row, col = edge_index
    args.user_dict = user_dict

    if args.model == 'CAGCN' or args.model == 'CAGCN-fusion':
        if args.type == 'jc':
            if args.dataset == 'libimseti':
                cal_trend = co_ratio_deg_user_jacard_sp
            else:
                cal_trend = co_ratio_deg_user_jacard
        elif args.type == 'co':
            if args.dataset == 'libimseti':
                cal_trend = co_ratio_deg_user_common_sp
            else:
                cal_trend = co_ratio_deg_user_common
        elif args.type == 'lhn':
            if args.dataset == 'libimseti':
                cal_trend = co_ratio_deg_user_lhn_sp
            else:
                cal_trend = co_ratio_deg_user_lhn
        elif args.type == 'sc':
            if args.dataset == 'libimseti':
                cal_trend = co_ratio_deg_user_sc_sp
            else:
                cal_trend = co_ratio_deg_user_sc

        path = os.getcwd() + '/dataset/' + args.dataset + \
            '/co_ratio_edge_weight_' + args.type + "_"+str(args.seed)+'.pt'

        if os.path.exists(path):
            trend = torch.load(path)
        else:
            print(args.dataset, 'calculate_CIR', 'count_time...')
            start = time.time()
            trend = cal_trend(
                adj_sp_norm, edge_index, deg, args)
            print('Preprocession', time.time() - start)

        norm = scatter_add(edge_weight, col, dim=0,
                           dim_size=args.n_users + args.n_items)[col]
        norm_now = scatter_add(
            trend, col, dim=0, dim_size=args.n_users + args.n_items)[col]

        if 'fusion' in args.model:
            trend = args.trend_coeff * trend / norm_now + edge_weight
        else:
            trend = args.trend_coeff * trend / norm_now * norm
        # trend = args.trend_coeff * trend / norm_now + edge_weight

        # visual_edge_weight(trend, edge_index, deg,
        #                    args.dataset, args.type)

        model.trend = (trend).to(args.device)
        args.model = args.model + '-' + args.type


    # load model from file

    if "fusion" in args.model:
        save_model_filename = args.path + '/trained_model/' + args.dataset + '/' + args.model + "_1.0_"+str(args.seed)+'.pkl'
    else:
        save_model_filename = args.path + '/trained_model/' + args.dataset + '/' + args.model + "_"+str(args.seed)+'.pkl'
    model.load_state_dict(torch.load(save_model_filename))
    model.eval()
    

    user_embs, item_embs = model.generate()

    print(user_embs.sum().item(), item_embs.sum().item())
    print(user_dict.keys())
    for k in user_dict.keys():
        s = 0
        for u in user_dict[k].keys():
            s += sum(user_dict[k][u])
        print(k, s)
    
    test_results = test(user_embs, item_embs, user_dict, args, flag="test")
    print(test_results)
    
    # obtain utility performance
    test_results = test(user_embs, item_embs, user_dict, args, flag="test")
    
    # obtain fairness performance
    fairness_measurement = test_fairness(user_embs, item_embs, user_dict, args, flag='test')

    save_list = [args.seed]
    for m in ['Recall', 'NDCG', 'Precision', 'Hit_ratio', 'F1']:
        save_list.append(test_results[m][3])
    save_list.extend(fairness_measurement)
    print(len(save_list))
    save_list = [str(t) for t in save_list]
    save_str = ' '.join(save_list)+"\n"
    
    save_filename = args.path+"/test_logs/"+args.dataset+"/"+args.model+".txt"
    save_file = open(save_filename, "a+")
    save_file.write(save_str)
    
