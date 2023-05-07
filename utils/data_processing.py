import logging
import time
import pickle as pk
import numpy as np
import pandas as pd

class Data:
    def __init__(self, data, is_split=False):
        self.srcs = data['src'].values
        self.dsts = data['dst'].values
        self.times = data['abs_time'].values
        self.trans_cascades = data['cas'].values
        self.pub_times = data['pub_time'].values
        self.labels = data['label'].values
        self.length = len(self.srcs)
        self.is_split = is_split
        if is_split:
            self.types = data['type'].values

    def loader(self, batch):
        for i in range(0, len(self.srcs), batch):
            right = min(i + batch, self.length)
            if self.is_split:
                yield (self.srcs[i:right], self.dsts[i:right], self.trans_cascades[i:right],
                       self.times[i:right], self.pub_times[i:right], self.types[i:right]), self.labels[i:right]
            else:
                yield (self.srcs[i:right], self.dsts[i:right], self.trans_cascades[i:right],
                       self.times[i:right], self.pub_times[i:right]), self.labels[i:right]


def get_label(x: pd.DataFrame, observe_time, label):
    id = np.searchsorted(x['time'], observe_time, side='left')
    casid = x['cas'].values[0]
    if casid in label and id >= 10:
        length = min(id, 100) - 1
        x['label'].iloc[length] = label[casid] - id
        return [x.iloc[:length + 1, :]]
    else:
        return []


def data_transformation(dataset, data, time_unit, min_time, param):
    if dataset == 'aps':
        data['pub_time'] = (pd.to_datetime(data['pub_time']) - pd.to_datetime(min_time)).apply(lambda x: x.days)
    else:
        data['pub_time'] -= min_time
    data['abs_time'] = (data['pub_time'] + data['time']) / time_unit
    data['pub_time'] /= time_unit
    data['time'] /= time_unit
    data.sort_values(by=['abs_time', 'id'], inplace=True, ignore_index=True)
    users = list(set(data['src']) | set(data['dst']))
    ids = list(range(len(users)))
    user2id, id2user = dict(zip(users, ids)), dict(zip(ids, users))
    cases = list(set(data['cas']))
    ids = list(range(len(cases)))
    cas2id, id2cas = dict(zip(cases, ids)), dict(zip(ids, cases))
    data['src'] = data['src'].apply(lambda x: user2id[x])
    data['dst'] = data['dst'].apply(lambda x: user2id[x])
    data['cas'] = data['cas'].apply(lambda x: cas2id[x])
    param['node_num'] = {'user': max(max(data['src']), max(data['dst'])) + 1, 'cas': max(data['cas']) + 1}
    param['max_global_time'] = max(data['abs_time'])
    pk.dump({'user2id': user2id, 'id2user': id2user, 'cas2id': cas2id, 'id2cas': id2cas},
            open(f'data/{dataset}_idmap.pkl', 'wb'))


def get_split_data(dataset, observe_time, predict_time, time_unit, all_data, min_time, metadata, log, param):
    def data_split(legal_cascades, train_portion=0.7, val_portion=0.15):
        """
        set cas type, 1 for train cas, 2 for val cas, 3 for test cas , and 0 for other cas that will be dropped
        """
        m_metadata = metadata[metadata['casid'].isin(set(legal_cascades))]
        all_idx, type_map = {}, {}
        if dataset == 'twitter':
            dt = pd.to_datetime(m_metadata['pub_time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
            idx = dt.apply(lambda x: not (x.month == 4 and x.day > 10)).values
        elif dataset == 'weibo':
            dt = pd.to_datetime(m_metadata['pub_time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
            idx = dt.apply(lambda x: x.hour < 18 and x.hour >= 8).values
        elif dataset == 'aps':
            idx = pd.to_datetime(m_metadata['pub_time']).apply(lambda x: x.year <= 1997).values
        else:
            idx = np.array([True] * len(m_metadata))
        cas = m_metadata[idx]['casid'].values
        rng = np.random.default_rng(0)
        rng.shuffle(cas)
        train_pos, val_pos = int(train_portion * len(cas)), int((train_portion + val_portion) * len(cas))
        train_cas, val_cas, test_cas = np.split(cas, [train_pos, val_pos])
        all_idx['train'] = train_cas
        type_map.update(dict(zip(train_cas, [1] * len(train_cas))))
        all_idx['val'] = val_cas
        type_map.update(dict(zip(val_cas, [2] * len(val_cas))))
        all_idx['test'] = test_cas
        type_map.update(dict(zip(test_cas, [3] * len(test_cas))))
        reset_cas = set(metadata['casid']) - set(train_cas) - set(val_cas) - set(test_cas)
        type_map.update(dict(zip(list(reset_cas), [0] * len(reset_cas))))
        return all_idx, type_map

    all_label = all_data[all_data['time'] < predict_time * time_unit].groupby(by='cas', as_index=False)['id'].count()
    all_label = dict(zip(all_label['cas'], all_label['id']))
    m_data = []
    for cas, df in all_data.groupby(by='cas'):
        m_data.extend(get_label(df, observe_time * time_unit, all_label))
    all_data = pd.concat(m_data, axis=0)
    all_idx, type_map = data_split(all_data[all_data['label'] != -1]['cas'].values)
    all_data['type'] = all_data['cas'].apply(lambda x: type_map[x])
    all_data = all_data[all_data['type'] != 0]
    """all_idx is used for baselines to select the cascade id, so it don't need to be remapped"""
    data_transformation(dataset, all_data, time_unit, min_time, param)
    all_data.to_csv(f'data/{dataset}_split.csv', index=False)
    pk.dump(all_idx, open(f'data/{dataset}_idx.pkl', 'wb'))
    log.info(
        f"Total Trans num is {len(all_data)}, Train cas num is {len(all_idx['train'])}, "
        f"Val cas num is {len(all_idx['val'])}, Test cas num is {len(all_idx['test'])}")
    return Data(all_data, is_split=True)


def get_data(dataset, observe_time, predict_time, train_time, val_time, test_time, time_unit,
             log: logging.Logger, param):
    a = time.time()
    """
    data stores all diffusion behaviors, in the form of (id,src,dst,cas,time). The `id` refers to the
    id of the interaction; `src`,`dst`,`cas`,`time` means that user `dst` forwards the message `cas` from `dst`
    after `time` time has elapsed since the publication of cascade `cas`. 
    -----------------
    metadata stores the metadata of cascades, including the publication time, publication user, etc.
    """
    data: pd.DataFrame = pd.read_csv(f'data/{dataset}.csv')
    metadata = pd.read_csv(f'data/{dataset}_metadata.csv')
    min_time = min(metadata['pub_time'])
    data = pd.merge(data, metadata, left_on='cas', right_on='casid')
    data = data[['id', 'src', 'dst', 'cas', 'time', 'pub_time']]
    param['max_time'] = {'user': 1, 'cas': param['observe_time']}
    data['label'] = -1
    data.sort_values(by='id', inplace=True, ignore_index=True)
    log.info(
        f"Min time is {min_time}, Train time is {train_time}, Val time is {val_time}, Test time is {test_time}, Time unit is {time_unit}")
    return_data = get_split_data(dataset, observe_time, predict_time, time_unit, data, min_time,
                                 metadata, log, param)
    b = time.time()
    log.info(f"Time cost for loading data is {b - a}s")
    return return_data
