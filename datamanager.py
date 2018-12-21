# coding:utf-8
'''
    datamanager for Air-writing dataset
'''
import numpy as np 
from utils import get_class_type, shuffle_in_unison_scary, padding
class datamanager(object):
    '''
        datamanager for loading 6DMG data.
        each .mat file has size of [14, T], where T is timesteps.
        starting from 0, 
            1 to 3 : x, y, z
            8 to 10: Ax, Ay, Az
            11 to 13: Gx, Gy, Gz
    '''
    def __init__(self, time_major=True, expand_dim=False, train_ratio=None, fold_k=None, seed=19940610):
        dataset = np.load("/home/scw4750/songbinxu/autoencoder/data/data.npz")
        # dataset = np.load("/home/scw4750/songbinxu/autoencoder/data/data_sampled.npz")
        self.fs = dataset["filenames"]
        self.AccGyo = dataset["AccGyo"]
        self.XYZ = dataset["XYZ"]
        self.labels = dataset["labels"]
        del dataset

        self.seed = seed
        self.divide_train_test(train_ratio, fold_k, seed=seed)

        self.train_cur_pos, self.test_cur_pos = 0, 0

        self.time_major = time_major
        self.expand_dim = expand_dim

    def change_test_fold(self, new_i):
        self.test_fold_id = new_i

    def get_train_test_id(self, train_ratio, fold_k, seed=None):
        self.train_id, self.test_id = [], []
        # normal
        if train_ratio and not fold_k:
            for v in self.dict_by_class:
                self.train_id += v[:int(train_ratio * len(v))]
                self.test_id += v[int(train_ratio * len(v)):]
        # cross validation
        if fold_k and not train_ratio:
            for i in range(62):
                self.test_id += list(self.dict_by_class[i][self.test_fold_id])
                for j in range(fold_k):
                    if j != self.test_fold_id:
                        self.train_id += list(self.dict_by_class[i][j])
        self.train_id = np.array(self.train_id)
        self.test_id = np.array(self.test_id)

        shuffle_in_unison_scary(self.train_id, self.test_id, seed=(seed or self.seed))
        self.train_num, self.test_num = len(self.train_id), len(self.test_id)

    def divide_train_test(self, train_ratio, fold_k, seed=None):
        # either train_ratio or fold_k must be None
        self.dict_by_class = [[] for i in range(62)]

        for i, key in enumerate(np.argmax(self.labels, axis=1)):
            self.dict_by_class[key].append(i)
        for i in range(62):
            np.random.seed(i)
            np.random.shuffle(self.dict_by_class[i])
        
        if fold_k and not train_ratio:
            # only for cross validation
            print "[{} folds cross validation]".format(fold_k)
            for i in range(62):
                np.random.seed(i)
                np.random.shuffle(self.dict_by_class[i])
                self.dict_by_class[i] = np.array_split(self.dict_by_class[i], fold_k)
                np.random.seed(i)
                np.random.shuffle(self.dict_by_class[i])
            self.test_fold_id = 0
        self.get_train_test_id(train_ratio, fold_k, seed)
    
    def shuffle_train(self, seed=None):
        np.random.seed(seed)
        np.random.shuffle(self.train_id)
    
    def shuffle_test(self, seed=None):
        np.random.seed(seed)
        np.random.shuffle(self.test_id)

    def get_cur_pos(self, cur_pos, full_num, batch_size):

        get_pos = range(cur_pos, cur_pos + batch_size)
        if cur_pos + batch_size <= full_num:
            cur_pos += batch_size
        else:
            rest = cur_pos + batch_size - full_num
            get_pos = range(cur_pos, full_num) + range(rest)
            cur_pos = rest
        return cur_pos, get_pos

    def __call__(self, batch_size, phase='train', maxlen=None, var_list=[]):
        if phase == 'train':
            self.train_cur_pos, get_pos = self.get_cur_pos(self.train_cur_pos, self.train_num, batch_size)
            cur_id = self.train_id[get_pos]
        elif phase == 'test':
            self.test_cur_pos, get_pos = self.get_cur_pos(self.test_cur_pos, self.test_num, batch_size)
            cur_id = self.test_id[get_pos]

        def get_closest_maxlen(L):
            tmp = [((i*2+11)*2+12)*2+12 for i in range(1, 23)] + [256]
            idx = np.argmin(np.abs(np.array(tmp) - L))
            return tmp[idx+1]

        def func(flag, maxlen=maxlen):
            if flag == 'lens':
                lens = np.array(map(len, self.AccGyo[cur_id]))
                lens = [get_closest_maxlen(L) for L in lens]
                return lens
            elif flag == 'AccGyo' or flag == 'XYZ':
                res = self.__dict__[flag][cur_id]
                maxlen = maxlen or max(map(len, res))
                maxlen = get_closest_maxlen(maxlen)
                res = padding(res, maxlen)
                if self.time_major:
                    res = np.transpose(res, (1,0,2))
                if self.expand_dim:
                    res = np.expand_dims(res, -1)
                return res
            elif flag == 'labels':
                return self.labels[cur_id]
            elif flag == 'filenames':
                return self.fs[cur_id]
        
        res = {}
        for key in var_list:
            if not res.has_key(key):
                res[key] = func(key)
        
        return res
