import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import time

def load_result(method_name, seed, topk_index, flag='reweight'):
    if flag == "reweight":
        prefix = "./test_logs_reweight/libimseti/"
    else:
        prefix = "./test_logs/libimseti/"
    rating_filename = prefix+method_name+"_"+str(seed)+"_rating_list.npy"
    rating_value_filename = prefix+method_name+"_"+str(seed)+"_rating_value_list.npy"
    high_rating_items = np.load(rating_filename, allow_pickle=True)
    ratings = np.load(rating_value_filename, allow_pickle=True)
    
    file_f1 = prefix+method_name+"_"+str(seed)+"_F1.npy"
    file_NDCG = prefix+method_name+"_"+str(seed)+"_NDCG.npy"
    file_hit_ratio = prefix+method_name+"_"+str(seed)+"_Hit_ratio.npy"
    file_recall = prefix+method_name+"_"+str(seed)+"_Recall.npy"
    file_precision = prefix+method_name+"_"+str(seed)+"_Precision.npy"
    file_user = prefix+method_name+"_"+str(seed)+"_user_ids.npy"
    
    f1 = np.load(file_f1, allow_pickle=True)[:, topk_index]
    ndcg = np.load(file_NDCG, allow_pickle=True)[:, topk_index]
    hit_ratio = np.load(file_hit_ratio, allow_pickle=True)[:, topk_index]
    recall = np.load(file_recall, allow_pickle=True)[:, topk_index]
    precision = np.load(file_precision, allow_pickle=True)[:, topk_index]
    results = dict()
    results['F1'] = f1
    results['NDCG'] = ndcg
    results['Hit_ratio'] = hit_ratio
    results['Recall'] = recall
    results['Precision'] = precision
    
    rec_user_list = np.load(file_user, allow_pickle=True)
    return high_rating_items, ratings, results, rec_user_list

def obtain_user_dict(seed):
    train_f = "./dataset/libimseti/processed_"+str(seed)+"/train_edges.txt"
    val_f = "./dataset/libimseti/processed_"+str(seed)+"/val_edges.txt"
    test_f = "./dataset/libimseti/processed_"+str(seed)+"/test_edges.txt"
    train_cf = np.loadtxt(open(train_f, "r"))
    val_cf = np.loadtxt(open(val_f, "r"))
    test_cf = np.loadtxt(open(test_f, "r"))
    train_data = train_cf.astype(int)
    val_data = val_cf.astype(int)
    test_data = test_cf.astype(int)
    
    n_users = max(max(train_data[:, 0]), max(val_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(val_data[:, 1]), max(test_data[:, 1])) + 1

    train_user_set, val_user_set, test_user_set, train_item_set = defaultdict(
        list), defaultdict(list), defaultdict(list), defaultdict(list)

    train_data[:, 1] += n_users
    val_data[:, 1] += n_users
    test_data[:, 1] += n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
        train_item_set[int(i_id)].append(int(u_id))
    for u_id, i_id in val_data:
        val_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
        
    user_dict = {
        'train_user_set': train_user_set,
        'val_user_set': val_user_set,
        'test_user_set': test_user_set,
        'train_item_set': train_item_set,
    }
    return user_dict, n_users


def Hit_at_k(r, k):
    right_pred = r[:, :k].sum(axis=1)

    return 1. * (right_pred > 0)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    # print(right_pred, 2213123213)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    precis = right_pred / precis_n
    return {'Recall': recall, 'Precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """

    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix

    # print(max_r[0], pred_data[0])
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)

    idcg[idcg == 0.] = 1.  # it is OK since when idcg == 0, dcg == 0
    ndcg = dcg / idcg
    # ndcg[np.isnan(ndcg)] = 0.

    return ndcg

def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def train_gender_distribution_from_train_edges(file_train_edges, id_to_gender_target_dict):
    train_edges = np.loadtxt(open(file_train_edges, "r")).astype(int)
#     print(train_edges)
    
    edgelist = defaultdict(list)
    for [u, v] in train_edges:
        edgelist[u].append(v)
#     print(edgelist)

    female_ratio = dict()
    for u in edgelist.keys():
        v_list = edgelist[u]
        v_gender = [id_to_gender_target_dict[t] for t in v_list]
#         print(v_gender)
        females = [int(t == 'F') for t in v_gender] # opposite gender: 1, same gender: 0
        female_ratio[u] = np.mean(females)
    return female_ratio


def split_groups_horizontally(filename, group_num=10):
# equal number range of ratio in different groups

    ratio_range = np.linspace(0, 1, group_num+1)

    # load opposite gender ratio
    id_to_ratio = dict()
    f = open(filename, "r")
    for line in f.readlines():
        line_split = line.rstrip().split()
#         print(line_split)
        uid = int(line_split[0])
#         print(uid)
        ratio = float(line_split[1])
        id_to_ratio[uid] = ratio

    # split
    groups = []
    for i, v in enumerate(ratio_range[:-1]):
        start_value = v
        end_value = ratio_range[i+1]
#         print(start_value, end_value)
        groups.append([k for k, v in filter (lambda x: x[1] >= start_value and x[1] < end_value, id_to_ratio.items())])
    final_end_value = ratio_range[-1]
    groups[-1].extend([k for k, v in filter (lambda x: x[1] == final_end_value, id_to_ratio.items())])
    group_size = [len(t) for t in groups]
    total_node = sum(group_size)
#     print(len(id_to_ratio), total_node)
    print(group_size)
#     for i, g in enumerate(groups):
#         if len(g) == 0:
#             print(i, "size 0")
#         else:
#             ratios = [id_to_ratio[uid] for uid in g]
#             print(i, min(ratios), max(ratios))
    return groups
    
def fair_measurement(group_scores):
    a = []
    for g0 in range(len(group_scores)):
        v0 = group_scores[g0]
        for g1 in range(g0+1, len(group_scores)):
            v1 = group_scores[g1]
            s = abs(v0-v1)
            a.append(s)
    f = np.mean(a)

    # update fairness by dividing the average performance for some kind of normalization
    avg_score = np.mean(group_scores)
    f = f/avg_score
        
    return f

def old_fair_measurement(group_scores):
    a = []
    for g0 in range(len(group_scores)):
        v0 = group_scores[g0]
        for g1 in range(g0+1, len(group_scores)):
            v1 = group_scores[g1]
            s = abs(v0-v1)
            a.append(s)
    f = np.mean(a)
    return f

def recommendation_gender_distribution(recommendation_list, id_to_gender_target_dict, topk=20):
    female_ratio = dict()
    for u in range(len(recommendation_list)):
        v_list = recommendation_list[u][:topk]
        v_gender = [id_to_gender_target_dict[t] for t in v_list]
        females = [int(t == 'F') for t in v_gender] 
        female_ratio[u] = np.mean(females)
    return female_ratio
    
def inconsistency_per_seed(seed, recommendation_list, id_to_gender_target_dict):   
    file_edges = "./dataset/libimseti/processed_"+str(seed)+"/train_edges.txt"
    file_id_to_gender_target = "./dataset/libimseti/processed_"+str(seed)+"/id_to_gender_target.npy"
    file_train_opposite_gender_ratio = "./dataset/libimseti/processed_"+str(seed)+"/opposite_gender_ratio.txt"

    # obtain train female ratio
    id_to_gender_target = np.load(file_id_to_gender_target, allow_pickle=True)
    id_to_gender_target_dict = dict()
    for k in id_to_gender_target.item().keys():
        id_to_gender_target_dict[k] = id_to_gender_target.item().get(k)

    # train_dict = train_gender_distribution_from_train_edges(file_train_edges, id_to_gender_target_dict)
    train_dict = train_gender_distribution_from_train_edges(file_edges, id_to_gender_target_dict)

    recommendation_dict = recommendation_gender_distribution(recommendation_list, id_to_gender_target_dict)

    groups = split_groups_horizontally(file_train_opposite_gender_ratio, group_num=3)
    group_inconsistency = []
    for i, g in enumerate(groups):
        t = []
        for uid in g:
            rec = recommendation_dict[uid]
            train = train_dict[uid]
            difference = abs(rec-train)
            t.append(difference)
        group_inconsistency.append(np.mean(t))
    return group_inconsistency



def absolute(u, v):
    return abs(u-v)

def selection_criterion(u_ratio, existing_gender, candidate_items, candidate_gender, candidate_rating, lambda_):
    # return the index of selected candidate
    if len(existing_gender) == 0:
        return 0
    
    female_num = np.sum(np.array([u == 'F' for u in existing_gender]).astype(int))
    
    max_score = -10000
    max_index = 0
    for i in range(len(candidate_items)):
        c_gender = candidate_gender[i]
        if c_gender == 'F':
            v_ratio = (1+female_num)/(len(existing_gender)+1)
        else:
            v_ratio = female_num/(len(existing_gender)+1)
        
        f = absolute(u_ratio, v_ratio)
        r = candidate_rating[i]
        score = (1-lambda_)*r - (lambda_)*f
        
        if score>max_score:
            max_score = score
            max_index = i
        
        if 'F' in candidate_gender[0:i+1] and 'M' in candidate_gender[0: i+1]: # both visit 'F' and 'M'
            # high r will be larger than low r with the same gender, therefore, once visited both gender, we can break
            break
    return max_index


class rerank():
    def __init__(self, method_name, seed, flag):
        self.user_dict, self.n_user = obtain_user_dict(seed=seed)
        self.seed = seed

        file_train_edges = "./dataset/libimseti/processed_"+str(seed)+"/train_edges.txt"
        file_val_edges = "./dataset/libimseti/processed_"+str(seed)+"/val_edges.txt"
        file_test_edges = "./dataset/libimseti/processed_"+str(seed)+"/test_edges.txt"
        file_train_opposite_gender_ratio = "./dataset/libimseti/processed_"+str(seed)+"/opposite_gender_ratio.txt"
        file_id_to_gender_src = "./dataset/libimseti/processed_"+str(seed)+"/id_to_gender_src.npy"
        file_id_to_gender_target = "./dataset/libimseti/processed_"+str(seed)+"/id_to_gender_target.npy"

        id_to_gender_src = np.load(file_id_to_gender_src, allow_pickle=True)
        self.id_to_gender_src_dict = dict()
        for k in id_to_gender_src.item().keys():
            self.id_to_gender_src_dict[k] = id_to_gender_src.item().get(k)

        id_to_gender_target = np.load(file_id_to_gender_target, allow_pickle=True)
        self.id_to_gender_target_dict = dict()
        for k in id_to_gender_target.item().keys():
            self.id_to_gender_target_dict[k] = id_to_gender_target.item().get(k)
            
        self.recommendation_list, self.recommendation_ratings, results, rec_user_list = load_result(method_name=method_name, seed=seed, topk_index=3, flag=flag)
        self.train_dict = train_gender_distribution_from_train_edges(file_train_edges, self.id_to_gender_target_dict)
        
    def rerank(self):
        results_selective = dict()
        results_selective_val = dict()
        items_lists_selective = dict() # record recommended items
        inconsistencies = dict()
        for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            r, u, list_ = self.evaluate_recommendation_selective(lambda_=t, flag="test")
            results_selective[t] = r
            items_lists_selective[t] = list_
            inconsistencies[t] = inconsistency_per_seed(self.seed, list_, self.id_to_gender_target_dict)
            print(str(t)+":", "Inconsistency =", np.array(inconsistencies[t]))

        return results_selective, inconsistencies
            
    # utils
    def rerank_selective(self, topk, lambda_, users):
        train_dict = self.train_dict
        id_to_gender_src_dict = self.id_to_gender_src_dict
        id_to_gender_target_dict = self.id_to_gender_target_dict
        rec_items = self.recommendation_list
        rec_ratings = self.recommendation_ratings
        
        recommendation_rerank = []
        for uid in users:
            uid = uid.item()
            female_ratio_train = train_dict[uid]
            rec_items_t = rec_items[uid][:topk]
            rec_gender = [id_to_gender_target_dict[u] for u in rec_items_t]
            female_ratio_rec = np.mean([t == 'F' for t in rec_gender])

            items_gender = []
            candidates = list(rec_items[uid])
            candidates_gender = [id_to_gender_target_dict[u] for u in candidates]
            u_ratings = list(rec_ratings[uid])
            # normalize the ratings to [0, 1]
            max_r = max(u_ratings)
            min_r = min(u_ratings)
            u_ratings = [(t - min_r)/(max_r - min_r) for t in u_ratings]

            u_recommendation_rerank = []
            for _ in range(topk):
                selected_item = selection_criterion(female_ratio_train, items_gender, candidate_items=candidates, \
                    candidate_gender=candidates_gender, candidate_rating=u_ratings, lambda_=lambda_)
                # selected item is the index in the candidates set
                u_recommendation_rerank.append(candidates[selected_item])
                items_gender.append(candidates_gender[selected_item])
                del candidates[selected_item]
                del candidates_gender[selected_item]
                del u_ratings[selected_item]
            recommendation_rerank.append(u_recommendation_rerank)
        return np.array(recommendation_rerank)

    def evaluate_recommendation_selective(self, lambda_, flag='test'):
        # need to obtain per user performance to calculate fairness
        train_dict = self.train_dict
        user_dict = self.user_dict
        id_to_gender_src_dict = self.id_to_gender_src_dict
        id_to_gender_target_dict = self.id_to_gender_target_dict
        n_users = self.n_user
        
        train_user_set = user_dict['train_user_set']
        val_user_set = user_dict['val_user_set']

        if flag == "test":
            test_user_set = user_dict['test_user_set']
            test_users = torch.tensor(list(test_user_set.keys()))
        elif flag == "val":
            test_user_set = user_dict['val_user_set']
            test_users = torch.tensor(list(test_user_set.keys()))

        recalls_ = []
        precisions_ = []
        f1s_ = []
        ndcgs_ = []
        hits_ = []

        with torch.no_grad():
            users_list = []
            ratings_list = []
            ratings_values_list = []
            groundTruth_items_list = []

            for batch_users in minibatch(test_users, batch_size=1024):
                if flag == "val":
                    clicked_items = [train_user_set[user.item()] - n_users
                                 for user in batch_users]
                elif flag == "test":
                    clicked_items = []
                    for user in batch_users:
                        clicked_items_for_user_in_train = train_user_set[user.item()] - n_users
                        clicked_items_for_user_in_val = val_user_set[user.item()] - n_users
                        clicked_items.append(np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))

                groundTruth_items = [test_user_set[user.item()] - n_users for user in batch_users]

                exclude_index = []
                exclude_items = []

                for range_i, items in enumerate(clicked_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)

                rating_K = self.rerank_selective(lambda_=lambda_, topk=20, users=batch_users)

                users_list.append(batch_users)
                ratings_list.append(rating_K)
                groundTruth_items_list.append(groundTruth_items)

            X = zip(ratings_list, groundTruth_items_list)

            for i, x in enumerate(X):
                sorted_items = x[0] #[1024, 100]
                groundTrue = x[1]
                r = getLabel(groundTrue, sorted_items)

                # print("shapes:", sorted_items.shape)
                # print(len(groundTrue)) # [1024]
                for iter_ in range(len(groundTrue)):
                    pre, recall, ndcg, hit_ratio, F1 = [], [], [], [], []
                    for k in [20]:
                        ret = RecallPrecision_ATk([groundTrue[iter_]], r[iter_].reshape(1, -1), k)
                        ndcgs = NDCGatK_r([groundTrue[iter_]], r[iter_].reshape(1, -1), k)
                        hit_ratios = Hit_at_k(r[iter_].reshape(1, -1), k)

                        hit_ratio.append(sum(hit_ratios))
                        pre.append(sum(ret['Precision']))
                        recall.append(sum(ret['Recall']))
                        ndcg.append(sum(ndcgs))

                        temp = ret['Precision'] + ret['Recall']
                        temp[temp == 0] = float('inf')
                        F1s = 2 * ret['Precision'] * ret['Recall'] / temp

                        F1.append(sum(F1s))

                    recalls_.append(np.array(recall))
                    precisions_.append(np.array(pre))
                    f1s_.append(np.array(F1))
                    ndcgs_.append(np.array(ndcg))
                    hits_.append(np.array(hit_ratio))
        results = dict()
        results['F1'] = np.array(f1s_)
        results['NDCG'] = np.array(ndcgs_)
        results['Hit_ratio'] = np.array(hits_)
        results['Recall'] = np.array(recalls_)
        results['Precision'] = np.array(precisions_)

        ratings_list = np.concatenate(ratings_list, axis=0)
        return results, users_list, np.array(ratings_list)

def main(method_name='MF', flag='no_reweight'):
    rerank_results_total = dict()
    for lambda_ in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rerank_results_total[lambda_] = {'Recall': [], 'Precision':[], 'F1':[], 'Hit_ratio':[], 'NDCG':[]}

    for seed in [1, 2, 3, 4, 5]:
        start = time.time()
        reranker = rerank(method_name=method_name, seed=seed, flag=flag)
        rerank_results = reranker.rerank()
        end = time.time()
        print("Total time:", end-start, "s")

        user_performance, inconsistency_performance = rerank_results[0], rerank_results[1]
        # user performance is a dict of lambdas and than the dict of F1, Recall...
        for l in user_performance.keys():
            for metric in ['Recall', 'Precision', 'F1', 'Hit_ratio', 'NDCG']:
                rerank_results_total[l][metric].append(user_performance[l][metric].squeeze())

    if flag == "reweight":
        save_path = './rerank_results_reweight/'
    else:
        save_path = './rerank_results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path, rerank_results_total)

main(method_name='MF', flag='no_reweight')
main(method_name='MF', flag='reweight')
