import dgl
from collections import defaultdict
import torch
from copy import deepcopy

class HGraph:
    def __init__(self, num_user, num_cas):
        self.num_user = num_user
        self.num_cas = num_cas
        self.init()

    def insert(self, cascades, srcs, dsts, abs_times, pub_times):
        self.cas_batch = defaultdict(list)
        times = abs_times - pub_times
        for cas, src, dst, time, abs_time in zip(cascades, srcs, dsts, times, abs_times):
            self.user_neighbor['follower'][src].append(dst)
            self.user_time['follower'][src].append(abs_time)
            self.user_neighbor['follow'][dst].append(src)
            self.user_time['follow'][dst].append(abs_time)
            self.cascades[cas].insert(src, dst, time, abs_time)
            self.cas_batch[cas].append((src, dst))

    def get_cas_seq(self, cascades):
        users, times, valid_length = [], [], []
        for cas in cascades:
            user, time = self.cascades[cas].get_seq()
            users.append(torch.tensor(user, dtype=torch.long))
            times.append(torch.tensor(time))
            valid_length.append(len(user))
        return users, times, valid_length

    def get_cas_pub_time(self, cascades):
        cas_pub_times = []
        for cas in cascades:
            cas_pub_times.append(self.cascades[cas].get_pub_time())
        return cas_pub_times

    def get_cas_graph(self, cascades):
        graph_roots, graph_leafs = [], []
        for cas in cascades:
            graph_root, graph_leaf = self.cascades[cas].get_graph()
            graph_roots.append(graph_root)
            graph_leafs.append(graph_leaf)
        return dgl.batch(graph_roots), dgl.batch(graph_leafs)

    def batch_cas_info(self):
        return deepcopy(self.cas_batch)

    def init(self):
        self.user_neighbor = {'follow': defaultdict(list),
                              'follower': defaultdict(list),
                              'neighbor': defaultdict(list)}
        self.user_time = deepcopy(self.user_neighbor)
        self.cascades = defaultdict(lambda: Cascade())


class Cascade:
    def __init__(self):
        self.seq = []
        self.dag = []
        self.pub_time = 1000000
        self.cnt = 0
        self.node2id = dict()
        self.id2node = dict()
        self.node_times = []

    def insert(self, u, v, t, abs_time):
        self.seq.append((v, t))
        self.pub_time = min(self.pub_time, abs_time)
        if v in self.node2id:
            if u not in self.node2id:
                return
            if self.node_times[self.node2id[u]] >= self.node_times[self.node2id[v]]:
                return
        for x in [u, v]:
            if x not in self.node2id:
                self.node2id[x] = self.cnt
                self.id2node[self.cnt] = x
                self.cnt += 1
                self.node_times.append(t)
        self.dag.append((u, v, t))

    def get_graph(self):
        srcs, dsts, times = zip(*self.dag)
        nodes = list(set(srcs) | set(dsts))
        ids = list(range(len(nodes)))
        map_srcs = list(map(lambda x: self.node2id[x], srcs))
        map_dsts = list(map(lambda x: self.node2id[x], dsts))
        graph_leaf = dgl.graph((map_srcs, map_dsts))
        graph_leaf.ndata['id'] = torch.tensor(list(map(lambda x: self.id2node[x], ids)))
        graph_leaf.edata['time'] = torch.tensor(times, dtype=torch.float)
        graph_leaf = dgl.RemoveSelfLoop()(graph_leaf)
        graph_leaf.ndata['mask'] = torch.tensor(graph_leaf.out_degrees() == 0, dtype=torch.float).unsqueeze(dim=-1)

        graph_root = dgl.reverse(graph_leaf, share_ndata=True, share_edata=True)
        graph_root.ndata['mask'] = torch.tensor(graph_root.out_degrees() == 0, dtype=torch.float).unsqueeze(dim=-1)

        return graph_root, graph_leaf

    def get_seq(self):
        users, times = zip(*self.seq)
        return users, times

    def get_pub_time(self):
        return self.pub_time
