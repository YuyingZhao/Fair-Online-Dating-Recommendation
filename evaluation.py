import numpy as np
from utils import *
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
np.set_printoptions(precision=4)

def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')


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


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]

    r = getLabel(groundTrue, sorted_items)

    pre, recall, ndcg, hit_ratio, F1 = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs = NDCGatK_r(groundTrue, r, k)
        hit_ratios = Hit_at_k(r, k)

        hit_ratio.append(sum(hit_ratios))
        pre.append(sum(ret['Precision']))
        recall.append(sum(ret['Recall']))
        ndcg.append(sum(ndcgs))

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s = 2 * ret['Precision'] * ret['Recall'] / temp
        # F1s[np.isnan(F1s)] = 0

        F1.append(sum(F1s))

    return {'Recall': np.array(recall),
            'Precision': np.array(pre),
            'NDCG': np.array(ndcg),
            'F1': np.array(F1),
            'Hit_ratio': np.array(hit_ratio)}


def test(user_embs, item_embs, user_dict, args, flag='val'):
    results = {'Precision': np.zeros(len(args.topks)),
               'Recall': np.zeros(len(args.topks)),
               'NDCG': np.zeros(len(args.topks)),
               'Hit_ratio': np.zeros(len(args.topks)),
               'F1': np.zeros(len(args.topks))}

    train_user_set = user_dict['train_user_set']
    val_user_set = user_dict['val_user_set']

    if flag == "test":
        test_user_set = user_dict['test_user_set']
        test_users = torch.tensor(list(test_user_set.keys()))
    elif flag == "val":
        test_user_set = user_dict['val_user_set']
        test_users = torch.tensor(list(test_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            # to introduce novelty of the recommended items
            # not recommend clicked items
            if flag == "val":
                clicked_items = [train_user_set[user.item()] - args.n_users
                                 for user in batch_users]
            elif flag == "test":
                clicked_items = []
                for user in batch_users:
                    clicked_items_for_user_in_train = train_user_set[user.item()] - args.n_users
                    clicked_items_for_user_in_val = val_user_set[user.item()] - args.n_users
                    clicked_items.append(np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))

            groundTruth_items = [test_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            # print("Max index:", max(exclude_index), max(exclude_items))
            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users)
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, args.topks))

        for result in pre_results:
            results['Recall'] += result['Recall']
            results['Precision'] += result['Precision']
            results['NDCG'] += result['NDCG']
            results['F1'] += result['F1']
            results['Hit_ratio'] += result['Hit_ratio']

        results['Recall'] /= len(test_users)
        results['Precision'] /= len(test_users)
        results['NDCG'] /= len(test_users)
        results['F1'] /= len(test_users)
        results['Hit_ratio'] /= len(test_users)

    return results

    
def split_groups_horizontally(filename, group_num=10):
# equal number range of ratio in different groups

    ratio_range = np.linspace(0, 1, group_num+1)

    # load opposite gender ratio
    id_to_ratio = dict()
    f = open(filename, "r")
    for line in f.readlines():
        line_split = line.rstrip().split()
        uid = int(line_split[0])
        ratio = float(line_split[1])
        id_to_ratio[uid] = ratio

    groups = []
    for i, v in enumerate(ratio_range[:-1]):
        start_value = v
        end_value = ratio_range[i+1]
        groups.append([k for k, v in filter (lambda x: x[1] >= start_value and x[1] < end_value, id_to_ratio.items())])
    final_end_value = ratio_range[-1]
    groups[-1].extend([k for k, v in filter (lambda x: x[1] == final_end_value, id_to_ratio.items())])
    group_size = [len(t) for t in groups]
    total_node = sum(group_size)
    print(group_size)
    return groups

# group information is input from a file (as a step during the preprocess)
# dict key is user id and value is its opposite gender percent during the interactions
def test_group(user_embs, item_embs, user_dict, args):
    opposite_ratio_filename = args.path+"/dataset/"+args.dataset+"/processed_"+str(args.seed)+"/opposite_gender_ratio.txt"
    groups = split_groups_horizontally(opposite_ratio_filename, args.group_num)
    # print("Group number: ", len(groups))
    # print("User numbers in each group: ", [len(t) for t in groups])

    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    val_user_set = user_dict['val_user_set']

    test_users = torch.tensor(list(test_user_set.keys()))

    results_whole_groups = []
    for gids in groups:
        # update the test user for each group
        # the test user for each group is the intersection of test user set and group user set
        # calculate the metrics for the specific user ids in this group

        group_test_user_set = set(test_user_set.keys()).intersection(set(gids))
        group_test_user = torch.tensor(list(group_test_user_set))

        results = {'Precision': np.zeros(len(args.topks)),
                   'Recall': np.zeros(len(args.topks)),
                   'NDCG': np.zeros(len(args.topks)),
                   'Hit_ratio': np.zeros(len(args.topks)),
                   'F1': np.zeros(len(args.topks))}

        with torch.no_grad():
            users_list = []
            ratings_list = []
            groundTruth_items_list = []

            for batch_users in minibatch(group_test_user, batch_size=args.test_batch_size):
                batch_users = batch_users.to(args.device)
                rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

                clicked_items = []
                for user in batch_users:
                    clicked_items_for_user_in_train = train_user_set[user.item()] - args.n_users
                    clicked_items_for_user_in_val = val_user_set[user.item()] - args.n_users
                    clicked_items.append(np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))

                groundTruth_items = [test_user_set[user.item()] - args.n_users
                                     for user in batch_users]

                exclude_index = []
                exclude_items = []

                for range_i, items in enumerate(clicked_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)

                rating_batch[exclude_index, exclude_items] = -(1 << 10)

                rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

                users_list.append(batch_users)
                ratings_list.append(rating_K)
                groundTruth_items_list.append(groundTruth_items)

            X = zip(ratings_list, groundTruth_items_list)

            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, args.topks))

            for result in pre_results:
                results['Recall'] += result['Recall']
                results['Precision'] += result['Precision']
                results['NDCG'] += result['NDCG']
                results['F1'] += result['F1']
                results['Hit_ratio'] += result['Hit_ratio']

            # need to divide the number of group test users rather than the original test users
            results['Recall'] /= len(group_test_user)
            results['Precision'] /= len(group_test_user)
            results['NDCG'] /= len(group_test_user)
            results['F1'] /= len(group_test_user)
            results['Hit_ratio'] /= len(group_test_user)
        results_whole_groups.append(results)

    return results_whole_groups

def obtain_gender(filename):
    gender = dict()
    f = np.load(filename, allow_pickle=True)
    for u in f.item().keys():
        gender[u] = f.item().get(u)
    return gender

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):
    score_matrix = score_matrix.cpu().detach()
    # print("Score matric shape: ", score_matrix.shape)
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[int(edge[0]), int(edge[1])]))
        else:
            preds_pos.append(score_matrix[int(edge[0]), int(edge[1])])
        pos.append(1) # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[int(edge[0]), int(edge[1])]))
        else:
            preds_neg.append(score_matrix[int(edge[0]), int(edge[1])])
        neg.append(0) # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def fair_measurement(group_result):
    fairness = []
    for m in ['Recall', 'NDCG', 'Precision', 'Hit_ratio', 'F1']:
        group_scores = []
        for i in range(len(group_result)):
            v = group_result[i][m][3]
            group_scores.append(v)

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

        fairness.append(f)
    return fairness

def test_fairness(user_embs, item_embs, user_dict, args, flag="val"):
    group_num = 3
    opposite_ratio_filename = args.path+"/dataset/"+args.dataset+"/processed_"+str(args.seed)+"/opposite_gender_ratio.txt"
    groups = split_groups_horizontally(opposite_ratio_filename, group_num)

    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    val_user_set = user_dict['val_user_set']

    test_users = torch.tensor(list(test_user_set.keys()))

    results_whole_groups = []
    for gids in groups:
        # update the test user for each group
        # the test user for each group is the intersection of test user set and group user set
        # calculate the metrics for the specific user ids in this group

        results = {'Precision': np.zeros(len(args.topks)),
               'Recall': np.zeros(len(args.topks)),
               'NDCG': np.zeros(len(args.topks)),
               'Hit_ratio': np.zeros(len(args.topks)),
               'F1': np.zeros(len(args.topks))}

        train_user_set = user_dict['train_user_set']
        val_user_set = user_dict['val_user_set']

        if flag == "test":
            test_user_set = user_dict['test_user_set']
        elif flag == "val":
            test_user_set = user_dict['val_user_set']

        group_test_user_set = set(test_user_set.keys()).intersection(set(gids))
        group_test_user = torch.tensor(list(group_test_user_set))

        with torch.no_grad():
            users_list = []
            ratings_list = []
            groundTruth_items_list = []

            for batch_users in minibatch(group_test_user, batch_size=args.test_batch_size):
                batch_users = batch_users.to(args.device)
                rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

                # to introduce novelty of the recommended items
                # not recommend clicked items
                if flag == "val":
                    clicked_items = [train_user_set[user.item()] - args.n_users
                                     for user in batch_users]
                elif flag == "test":
                    clicked_items = []
                    for user in batch_users:
                        clicked_items_for_user_in_train = train_user_set[user.item()] - args.n_users
                        clicked_items_for_user_in_val = val_user_set[user.item()] - args.n_users
                        clicked_items.append(np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))

                groundTruth_items = [test_user_set[user.item()] - args.n_users
                                     for user in batch_users]

                exclude_index = []
                exclude_items = []

                for range_i, items in enumerate(clicked_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)

                rating_batch[exclude_index, exclude_items] = -(1 << 10)

                rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

                users_list.append(batch_users)
                ratings_list.append(rating_K)
                groundTruth_items_list.append(groundTruth_items)

            X = zip(ratings_list, groundTruth_items_list)

            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, args.topks))

            for result in pre_results:
                results['Recall'] += result['Recall']
                results['Precision'] += result['Precision']
                results['NDCG'] += result['NDCG']
                results['F1'] += result['F1']
                results['Hit_ratio'] += result['Hit_ratio']

            # need to divide the number of group test users rather than the original test users
            results['Recall'] /= len(group_test_user)
            results['Precision'] /= len(group_test_user)
            results['NDCG'] /= len(group_test_user)
            results['F1'] /= len(group_test_user)
            results['Hit_ratio'] /= len(group_test_user)
        results_whole_groups.append(results)

    # print("Results whole groups:")
    # print(results_whole_groups)

    return fair_measurement(results_whole_groups)


