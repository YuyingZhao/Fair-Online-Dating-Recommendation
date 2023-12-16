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


def run(weights, model, optimizer, train_cf, clicked_set, user_dict, adj, args):
    test_recall_best, early_stop_count = -float('inf'), 0
    utility_best, fairness_best = 0, 0
    test_recall_best_epoch = 0

    adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
    edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

    model.adj_sp_norm = adj_sp_norm.to(args.device)
    model.edge_index = edge_index.to(args.device)
    model.edge_weight = edge_weight.to(args.device)
    model.deg = deg.to(args.device)

    row, col = edge_index
    args.user_dict = user_dict

    if args.model == 'CAGCN':
        if args.type == 'jc':
            if args.dataset in ['libimseti']:
                cal_trend = co_ratio_deg_user_jacard_sp
            else:
                cal_trend = co_ratio_deg_user_jacard
        elif args.type == 'co':
            if args.dataset in ['libimseti']:
                cal_trend = co_ratio_deg_user_common_sp
            else:
                cal_trend = co_ratio_deg_user_common
        elif args.type == 'lhn':
            if args.dataset in ['libimseti']:
                cal_trend = co_ratio_deg_user_lhn_sp
            else:
                cal_trend = co_ratio_deg_user_lhn
        elif args.type == 'sc':
            if args.dataset in ['libimseti']:
                cal_trend = co_ratio_deg_user_sc_sp
            else:
                cal_trend = co_ratio_deg_user_sc

        path = os.getcwd() + '/dataset/' + args.dataset + \
            '/co_ratio_edge_weight_' + args.type + '_'+str(args.seed)+'.pt'

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

        trend = args.trend_coeff * trend / norm_now * norm

        model.trend = (trend).to(args.device)
        args.model = args.model + '-' + args.type

    losses, recall, ndcg, precision, hit_ratio, F1 = defaultdict(list), defaultdict(
        list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    start = time.time()

    total_losses = []



    for epoch in range(args.epochs):
        neg_cf = neg_sample_before_epoch(train_cf, clicked_set, args)

        dataset = Dataset(
            users=train_cf[:, 0], pos_items=train_cf[:, 1], neg_items=neg_cf, args=args)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                collate_fn=dataset.collate_batch, pin_memory=args.pin_memory)  # organzie the dataloader based on re-sampled negative pairs

        """training"""
        model.train()
        loss = 0

        batch_losses = []
        for i, batch in enumerate(dataloader):

            batch = batch_to_gpu(batch, args.device)
        
            user_embs, pos_item_embs, neg_item_embs, user_embs0, pos_item_embs0, neg_item_embs0 = model(batch)

            if args.model == "LightGCN" or 'CAGCN' in args.model:
                bpr_loss = cal_bpr_loss_light_gcn_reweight(weights[batch['users']], user_embs, pos_item_embs, neg_item_embs)
            else:
                bpr_loss = cal_bpr_loss_reweight(weights[batch['users']], user_embs, pos_item_embs, neg_item_embs)

            # l2 regularization
            l2_loss = cal_l2_loss(
                user_embs0, pos_item_embs0, neg_item_embs0, user_embs0.shape[0])
            batch_loss = bpr_loss + args.l2 * l2_loss
            batch_losses.append(batch_loss.item())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()

        #******************evaluation****************
        total_losses.append(loss / (i + 1))

        if not epoch % 10:
            model.eval()
            res = PrettyTable()
            res.field_names = ["Time", "Epoch", "Training_loss",
                               "Recall", "NDCG", "Precision", "Hit_ratio", "F1"]

            user_embs, item_embs = model.generate()
            test_res = test(user_embs, item_embs, user_dict, args, flag='val')
            res.add_row(
                [format(time.time() - start, '.4f'), epoch, format(loss / (i + 1), '.4f'), test_res['Recall'], test_res['NDCG'],
                 test_res['Precision'], test_res['Hit_ratio'], test_res['F1']])

            print(res)

            for k in args.topks:
                recall[k].append(test_res['Recall'])
                ndcg[k].append(test_res['NDCG'])
                precision[k].append(test_res['Precision'])
                hit_ratio[k].append(test_res['Hit_ratio'])
                F1[k].append(test_res['F1'])
                losses[k].append(loss / (i + 1))

            # *********************************************************
            # 3 relates to topk=20
            utility_measurement = (test_res['Recall'][3]+test_res['NDCG'][3]+test_res['Precision'][3]+\
                test_res['Hit_ratio'][3]+test_res['F1'][3])/5
            fairness_measurement = np.mean(test_fairness(user_embs, item_embs, user_dict, args, flag='val'))
            score = utility_measurement - fairness_measurement

            if score > test_recall_best:
                test_recall_best = score
                test_recall_best_epoch = epoch
                early_stop_count = 0
                utility_best = utility_measurement
                fairness_best = fairness_measurement

                if args.save:
                    torch.save(model.state_dict(), os.getcwd() +
                               '/trained_model_reweight/' + args.model + "_" + str(args.seed)\
                               + "_"+str(args.penalty)+'.pkl'
                               )

    print("Best validation: ", args.seed, args.penalty, test_recall_best, utility_best, fairness_best, \
        test_recall_best_epoch)

    return test_recall_best

def obtain_train_weight_power(args):
    # first obtain the opposite gender ratio
    # penalty is used to control the power here
    ogr_filename = args.path+"/dataset/"+args.dataset+"/processed_"+str(args.seed)\
    +"/opposite_gender_ratio.txt"

    groups = split_groups_horizontally(ogr_filename, group_num=3)
    user_num = [len(g) for g in groups]
    print("User number:", user_num)

    ogr_array = np.loadtxt(ogr_filename)

    denominator = 0
    for n in user_num:
        denominator += n**(1.0-args.penalty)
    T = len(ogr_array)*1.0/(denominator)
    group_weight = [T/(l**args.penalty) for l in user_num]
 
    w = np.array([0.0]*len(ogr_array))
    for i, g in enumerate(groups):
        w[g] = group_weight[i]
    
    w = list(w)
    return w

if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)
    print("Configurations:", args)

    """build dataset"""
    train_cf, val_cf, test_cf, user_dict, args.n_users, args.n_items, clicked_set, adj = load_data(args)

    print(args.n_users, args.n_items, train_cf.shape[0] + val_cf.shape[0], test_cf.shape[0],
        (train_cf.shape[0] + val_cf.shape[0] + test_cf.shape[0]) / (args.n_items * args.n_users))


    if(args.neg_in_val_test == 1):  # whether negative samples from validation and test sets
        clicked_set = user_dict['train_user_set']

    """build model"""
    if args.model == 'LightGCN':
        model = LightGCN(args).to(args.device)
    elif args.model == 'NGCF':
        model = NGCF(args).to(args.device)
    elif args.model == 'MF':
        model = MF(args).to(args.device)
    elif args.model == 'CAGCN':
        model = CAGCN(args).to(args.device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # assign the weight for optimization
    w = obtain_train_weight_power(args)
    w = torch.tensor(w).to(args.device)


    # create dictionaries if not exists (trained_model/dataset/)
    paths = [args.path+"/trained_model_reweight/", \
             args.path+"/trained_model_reweight/"+args.dataset, \
                args.path+"/test_logs_reweight"]
    for p in paths: 
        if not os.path.exists(p):
            os.mkdir(p)
            print("path has been created: ", p)

    best_val = run(w, model, optimizer, train_cf, clicked_set, user_dict, adj, args)

    # test
    save_model_filename = args.path+\
    '/trained_model_reweight/' + args.model + "_" + str(args.seed)+\
    "_"+str(args.penalty)+'.pkl'
    model.load_state_dict(torch.load(save_model_filename))
    model.eval()

    user_embs, item_embs = model.generate()

    # store test result for experiment result
    test_results = test(user_embs, item_embs, user_dict, args, flag='test')
    fairness_measurement = test_fairness(user_embs, item_embs, user_dict, args, flag='test')

    save_list = [args.seed, args.penalty]
    for m in ['Recall', 'NDCG', 'Precision', 'Hit_ratio', 'F1']:
        save_list.append(test_results[m][3])
    save_list.extend(fairness_measurement)
    save_list = [str(t) for t in save_list]
    save_str = ' '.join(save_list)+"\n"
    
    save_filename = args.path+"/test_logs_reweight/"+args.model+"_test_result.txt"
    save_file = open(save_filename, "a+")
    save_file.write(save_str)

    # store validation result for model selection
    test_results = test(user_embs, item_embs, user_dict, args, flag='val')
    fairness_measurement = test_fairness(user_embs, item_embs, user_dict, args, flag='val')

    save_list = [args.seed, args.penalty]
    for m in ['Recall', 'NDCG', 'Precision', 'Hit_ratio', 'F1']:
        save_list.append(test_results[m][3])
    save_list.extend(fairness_measurement)
    save_list = [str(t) for t in save_list]
    save_str = ' '.join(save_list)+"\n"
    
    save_filename = args.path+"/test_logs_reweight/"+args.model+"_val_full_result.txt"
    save_file = open(save_filename, "a+")
    save_file.write(save_str)

    # abbreviated results
    save_list = [args.seed, args.penalty, best_val]
    save_list = [str(t) for t in save_list]
    save_str = ' '.join(save_list)+"\n"
    best_validation_filename = args.path+"/test_logs_reweight/"+args.model+"_val_result.txt"

    best_validation_file = open(best_validation_filename, "a+")
    best_validation_file.write(save_str)



