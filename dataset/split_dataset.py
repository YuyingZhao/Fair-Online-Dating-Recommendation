import argparse
from collections import defaultdict
import numpy as np
import os
import random
import pandas

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="libimseti",
                        help="Choose a dataset:[libimseti]")
    parser.add_argument("--kcore_iteration_num", type=int, default=10,
        help="repeatedly remove nodes with interactions less than kcore number \
        until reaching convergence")
    parser.add_argument("--kcore", type=int, default=5,
        help="kcore number")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

class dataset_spliter():
    def __init__(self, args):
        self.kcore = args.kcore
        self.val_frac = args.val_frac
        self.test_frac = args.test_frac
        self.train_frac = 1 - self.test_frac - self.val_frac
        self.rating_threshold = args.rating_threshold
        self.dataset_path = args.path
        self.dataset = args.dataset
        self.save_flag_current_iteration = 0
        self.kcore_iteration_num = args.kcore_iteration_num

        self.edges = None
        self.old_to_new_src = None
        self.old_to_new_target = None

        self.uid_to_gender = None
        self.uid_to_gender_src = None
        self.uid_to_gender_target = None

    def load_gender(self):
        gender_filename = self.dataset_path+"/src/gender.txt"
        user_to_gender = dict() # key is user id, and value is F or M
        f = open(gender_filename, "r")
        for line in f.readlines():
            line_split = line.rstrip().split(",")
            user_id = int(line_split[0])
            gender = line_split[1]
            if gender != "U":
                user_to_gender[user_id] = gender
        return user_to_gender

    def save_new_uid_to_gender(self):
        # save the mapping from new_id (src and target) to gender
        new_to_old_src = {self.old_to_new_src[k]:k for k in self.old_to_new_src.keys()}
        new_to_old_target = {self.old_to_new_target[k]:k for k in self.old_to_new_target.keys()}

        edges = np.array(self.edges)
        src = edges[:, 0]
        target = edges[:, 1]

        new_uid_to_gender_src = dict()
        for n in src:
            old_id = new_to_old_src[n]
            new_uid_to_gender_src[n] = self.uid_to_gender[old_id]

        new_uid_to_gender_target = dict()
        for n in target:
            old_id = new_to_old_target[n]
            new_uid_to_gender_target[n] = self.uid_to_gender[old_id]

        if self.save_flag_current_iteration == 1:
            np.save(self.dataset_path + '/processed_'+str(args.seed)+'/id_to_gender_src.npy', new_uid_to_gender_src)
            np.save(self.dataset_path + '/processed_'+str(args.seed)+'/id_to_gender_target.npy', new_uid_to_gender_target)
            print("Finish saving gender information")
        return new_uid_to_gender_src, new_uid_to_gender_target

    def save_opposite_gender_ratio(self, train_edges):
        # save the OGIR into file

        # transform edge to adj list
        edgelist = defaultdict(list)
        for [u, v] in train_edges:
            edgelist[u].append(v)

        uid_to_ratio = dict()
        for u in edgelist.keys():
            u_gender = self.uid_to_gender_src[u]
            v_list = edgelist[u]
            v_gender = [self.uid_to_gender_target[t] for t in v_list]
            gender_diff = [int(u_gender != t) for t in v_gender] # opposite gender: 1, same gender: 0
            opposite_proportion = np.mean(gender_diff)
            uid_to_ratio[u] = opposite_proportion

        train_set = set(np.array(train_edges)[:, 0])

        if self.save_flag_current_iteration == 1:
            filename = self.dataset_path + '/processed_'+str(args.seed)+'/opposite_gender_ratio.txt'
            f = open(filename, "w")
            for k in uid_to_ratio.keys():
                if k in train_set:
                    f.write(str(k)+" "+str(uid_to_ratio[k])+"\n")
            f.close()
            print("Finish saving opposite gender ratio")

    def remove_duplicate(self, edges):
        edges = [(u, v) for u, v in edges]
        edges = set(edges)
        edges = list(edges)
        return edges

    def load_and_filter_interactions(self):
        # load information from edge file
        # save edges [u, v, r] where
        # r >= rating_threshold
        # where users have gender
        self.uid_to_gender = self.load_gender()

        user_with_gender_set = set(self.uid_to_gender.keys())
        edge_filename = self.dataset_path+"/src/edges.txt"
        edges = []
        f = open(edge_filename, "r")
        for line in f.readlines():
            line_split = line.rstrip().split()
            u, v, r = int(line_split[0]), int(line_split[1]), int(line_split[2])
            if u in user_with_gender_set and v in user_with_gender_set and r >= self.rating_threshold:
                edges.append([u, v])
        edges = self.remove_duplicate(edges)
        return np.array(edges)   

    def print_graph_information(self, edges=None):
        if edges is None:
            edges = self.edges
        print("Source node number:", len(set(edges[:, 0])),
            "Target Node number:", len(set(edges[:, 1])),
            "Edge number:", len(edges))

    def apply_kcore(self):
        # update self.edges to only include users that have at least kcore interactions
        # kcore setting to ensure the quality of dataset, default=10-core
        # which means that only users with at least 10 ratings are maintained

        # transform edge to adj list
        edgelist = defaultdict(list) # src to target
        edgelist2 = defaultdict(list) # target to src
        for [u, v] in self.edges:
            edgelist[u].append(v)
            edgelist2[v].append(u)

        src_list = []
        for k in edgelist.keys():
            if len(edgelist[k]) >= self.kcore:
                src_list.append(k)

        target_list = []
        for k in edgelist2.keys():
            if len(edgelist2[k]) >= self.kcore:
                target_list.append(k)

        new_edges = []
        src_set = set(src_list)
        target_set = set(target_list)
        for [u, v] in self.edges:
            if u in src_set and v in target_set:
                new_edges.append([u, v])
        return np.array(new_edges)

    def re_index(self):
        # re-index the node ids in the filtered dataset
        # return the updated edges and the node map for future use
        # the node_map is a map from old node id to new node id
        # id start from 0
        # reindex source and target seperately
        node_map_src = dict()
        node_map_target = dict()
        cnt_src = 0
        cnt_target = 0
        for u, v in self.edges:
            if u not in node_map_src.keys():
                node_map_src[u] = cnt_src
                cnt_src += 1
            if v not in node_map_target.keys():
                node_map_target[v] = cnt_target
                cnt_target += 1

        # transform
        new_edges = [[node_map_src[u], node_map_target[v]] for u, v in self.edges]

        return np.array(new_edges), node_map_src, node_map_target

    def seperate_gender(self, edge_list):
        interaction_list = edge_list[:, 1]
        genders = np.array([self.uid_to_gender[t] for t in interaction_list])
        indices = np.array(range(len(interaction_list))) # obtain the local index rather than the global one
        female = indices[genders=='F']
        male = indices[genders=='M']
        return female, male

    def split(self):
        self.edges = self.load_and_filter_interactions()
        self.print_graph_information()

        last_edge_num = 0
        for i in range(self.kcore_iteration_num):
            self.edges = self.apply_kcore()
            if last_edge_num == len(self.edges):
                self.save_flag_current_iteration = 1
                print("Kcore converges in:", i, "iteration")
                isExist = os.path.exists(self.dataset_path + '/processed_'+str(args.seed))
                if not isExist:
                   os.makedirs(self.dataset_path + '/processed_'+str(args.seed))
            else:
                last_edge_num = len(self.edges)

            print("\n-----------In kcore: iterating time "+str(i)+"-----------")
            print("After kcore:")
            self.print_graph_information()

            # split train/val/test
            # save into train/val/test_edges.txt
            # save into val_false/test_false_edges.txt

            # sort edges
            self.edges = np.array(self.edges)
            self.edges = self.edges[self.edges[:, 0].argsort()]

            train_edges, val_edges, test_edges = [], [], []
            head_0 = self.edges[0, 0]
            l, r = 0, 0
            while r < self.edges.shape[0]:
                if self.edges[r, 0] != self.edges[l, 0]:
                    tmp_edge_list = np.random.permutation(self.edges[l:r, :])
                    interaction_num = len(tmp_edge_list)

                    if interaction_num < self.kcore:
                        l = r
                        continue

                    # obtain female and male number in the interactions
                    female, male = self.seperate_gender(tmp_edge_list)
                    female_num = len(female)
                    female_train_size = int(female_num * self.train_frac)
                    female_test_size = int(female_num * self.test_frac)
                    female_val_size = female_num - female_train_size - female_test_size

                    train_size = int(interaction_num * self.train_frac)
                    test_size = int(interaction_num * self.test_frac)
                    val_size = interaction_num - train_size - test_size

                    male_train_size = train_size - female_train_size
                    male_test_size = test_size - female_test_size
                    male_val_size = val_size - female_val_size

                    train_edges.extend(tmp_edge_list[female[0: female_train_size]].tolist())
                    train_edges.extend(tmp_edge_list[male[0: male_train_size]].tolist())
                    val_edges.extend(tmp_edge_list[female[female_train_size:female_train_size+female_val_size]].tolist())
                    val_edges.extend(tmp_edge_list[male[male_train_size:male_train_size+male_val_size]].tolist())
                    test_edges.extend(tmp_edge_list[female[-female_test_size:]].tolist())
                    test_edges.extend(tmp_edge_list[male[-male_test_size:]].tolist())

                    l = r
                r += 1

            # record the last node's interactions
            tmp_edge_list = np.random.permutation(self.edges[l:r, :])
            interaction_num = len(tmp_edge_list)

            if interaction_num >= self.kcore:

                # obtain female and male number in the interactions
                female, male = self.seperate_gender(tmp_edge_list)
                female_num = len(female)
                female_train_size = int(female_num * self.train_frac)
                female_test_size = int(female_num * self.test_frac)
                female_val_size = female_num - female_train_size - female_test_size

                train_size = int(interaction_num * self.train_frac)
                test_size = int(interaction_num * self.test_frac)
                val_size = interaction_num - train_size - test_size

                male_train_size = train_size - female_train_size
                male_test_size = test_size - female_test_size
                male_val_size = val_size - female_val_size

                train_edges.extend(tmp_edge_list[female[0: female_train_size]].tolist())
                train_edges.extend(tmp_edge_list[male[0: male_train_size]].tolist())
                val_edges.extend(tmp_edge_list[female[female_train_size:female_train_size+female_val_size]].tolist())
                val_edges.extend(tmp_edge_list[male[male_train_size:male_train_size+male_val_size]].tolist())
                test_edges.extend(tmp_edge_list[female[-female_test_size:]].tolist())
                test_edges.extend(tmp_edge_list[male[-male_test_size:]].tolist())

            # only do reindex before saving, in this way, the old refers to the initial index
            if self.save_flag_current_iteration == 1:
                self.edges, self.old_to_new_src, self.old_to_new_target = self.re_index()
                # also save the index for matching purpose
                np.save(self.dataset_path + '/processed_'+str(args.seed)+'/old_to_new_src.npy', self.old_to_new_src)
                np.save(self.dataset_path + '/processed_'+str(args.seed)+'/old_to_new_target.npy', self.old_to_new_target)
                print("Finish saving index")
                # update the train/val/test according to the new index
                train_edges = [[self.old_to_new_src[u], self.old_to_new_target[v]] for u, v in train_edges]
                val_edges = [[self.old_to_new_src[u], self.old_to_new_target[v]] for u, v in val_edges]
                test_edges = [[self.old_to_new_src[u], self.old_to_new_target[v]] for u, v in test_edges]

            print("Train/Val/Test:")
            self.print_graph_information(np.array(train_edges))
            self.print_graph_information(np.array(val_edges))
            self.print_graph_information(np.array(test_edges))

            if self.save_flag_current_iteration == 1:
                np.savetxt(self.dataset_path + '/processed_'+str(args.seed)+'/train_edges.txt', train_edges, fmt='%i')
                np.savetxt(self.dataset_path + '/processed_'+str(args.seed)+'/val_edges.txt', val_edges, fmt='%i')
                np.savetxt(self.dataset_path + '/processed_'+str(args.seed)+'/test_edges.txt', test_edges, fmt='%i')
                print("Finish saving Train/Val/Test edges: ", i)

                self.uid_to_gender_src, self.uid_to_gender_target = self.save_new_uid_to_gender()
                self.save_opposite_gender_ratio(train_edges)

                break


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    args.path = os.getcwd() + "/" + args.dataset
    args.val_frac = 0.2
    args.test_frac = 0.2
    args.rating_threshold = 10

    print("Start splitting")
    print("Configuration:", args)

    spliter = dataset_spliter(args=args)
    spliter.split()

