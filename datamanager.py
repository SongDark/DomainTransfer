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
    def __init__(self, time_major=True, seed=19940610):
        dataset = np.load("/home/scw4750/songbinxu/autoencoder/data/data.npz")
        self.fs = dataset["filenames"]
        self.AccGyo = dataset["AccGyo"]
        self.XYZ = dataset["XYZ"]
        self.labels = dataset["labels"]
        del dataset

        self.divide_train_test(seed=seed)

        self.train_cur_pos, self.test_cur_pos = 0, 0

        self.time_major = time_major
    
    def divide_train_test(self, train_ratio=0.8, seed=None):
        self.dict_by_class = [[] for i in range(62)]

        for i, key in enumerate(np.argmax(self.labels, axis=1)):
            self.dict_by_class[key].append(i)
        
        self.train_id, self.test_id = [], []
        for i, v in enumerate(self.dict_by_class):
            np.random.seed(i)
            np.random.shuffle(v)
            self.train_id += v[:int(train_ratio * len(v))]
            self.test_id += v[int(train_ratio * len(v)):]
        
        self.train_id = np.array(self.train_id)
        self.test_id = np.array(self.test_id)
        
        if seed:
            shuffle_in_unison_scary(self.train_id, self.test_id, seed=seed)
        
        self.train_num, self.test_num = len(self.train_id), len(self.test_id)
    
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

    def __call__(self, batch_size, phase='train', mode='seq2seq', var_list=[]):
        if phase == 'train':
            self.train_cur_pos, get_pos = self.get_cur_pos(self.train_cur_pos, self.train_num, batch_size)
            cur_id = self.train_id[get_pos]
        elif phase == 'test':
            self.test_cur_pos, get_pos = self.get_cur_pos(self.test_cur_pos, self.test_num, batch_size)
            cur_id = self.test_id[get_pos]
        
        def func(flag):
            if flag == 'lens':
                return np.array(map(len, self.AccGyo[cur_id]))
            elif flag == 'AccGyo' or flag == 'XYZ':
                res = self.__dict__[flag][cur_id]
                maxlen = max(map(len, res))
                res = padding(res, maxlen)
                if self.time_major:
                    res = np.transpose(res, (1,0,2))
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
