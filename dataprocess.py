import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset as BaseDataset
from torch_geometric.utils import add_remaining_self_loops, degree

class Dataset(BaseDataset):
    def __init__(self, users, pos_items, neg_items, args):
        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items
        self.args = args
        # print("dataset number:", len(self.users))

    def _get_feed_dict(self, index):

        feed_dict = {
            'users': self.users[index],
            'pos_items': self.pos_items[index],
            'neg_items': self.neg_items[index],
        }

        return feed_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        # feed_dicts: [dict1, dict2, ...]

        feed_dict = dict()

        feed_dict['users'] = torch.LongTensor([d['users'] for d in feed_dicts])
        feed_dict['pos_items'] = torch.LongTensor(
            [d['pos_items'] for d in feed_dicts])

        feed_dict['neg_items'] = torch.LongTensor(
            np.stack([d['neg_items'] for d in feed_dicts]))

        feed_dict['idx'] = torch.cat(
            [feed_dict['users'], feed_dict['pos_items'], feed_dict['neg_items'].view(-1)])

        return feed_dict

def process(train_data, val_data, test_data):
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

    return n_users, n_items, train_user_set, val_user_set, test_user_set, train_item_set

def process_adj(data_cf, n_users, n_items):
    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1]  # [0, n_items) -> [n_users, n_users+n_items)
    # note this has been done in process, therefore, no need to do it here
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    return torch.LongTensor(cf_).t()

def load_data(args):
    print('reading train/val/test user-item set ...')
    train_f = args.path + '/dataset/' + args.dataset + '/processed_'+str(args.seed)+'/train_edges.txt'
    val_f = args.path + '/dataset/' + args.dataset + '/processed_'+str(args.seed)+'/val_edges.txt'
    test_f = args.path + '/dataset/' + args.dataset + '/processed_'+str(args.seed)+'/test_edges.txt'
    train_cf = np.loadtxt(open(train_f, "r"))
    val_cf = np.loadtxt(open(val_f, "r"))
    test_cf = np.loadtxt(open(test_f, "r"))
    train_cf = train_cf.astype(int)
    val_cf = val_cf.astype(int)
    test_cf = test_cf.astype(int)
    print("Train: ", train_cf.shape)
    print("Val: ", val_cf.shape)
    print("Test: ", test_cf.shape)

    n_users, n_items, train_user_set, val_user_set, test_user_set, train_item_set = process(train_cf, val_cf, test_cf)

    print('building the adj mat ...')
    adj = process_adj(train_cf, n_users, n_items)

    user_dict = {
        'train_user_set': train_user_set,
        'val_user_set': val_user_set,
        'test_user_set': test_user_set,
        'train_item_set': train_item_set,
    }

    clicked_set = defaultdict(list)
    for key in user_dict:
        for user in user_dict[key]:
            clicked_set[user].extend(user_dict[key][user])

    print('Finish loading dataset', args.dataset)
    return train_cf, val_cf, test_cf, user_dict, n_users, n_items, clicked_set, adj


def normalize_edge(edge_index, n_users, n_items):
    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse.FloatTensor(edge_index, edge_weight, (n_users + n_items, n_users + n_items)), deg
