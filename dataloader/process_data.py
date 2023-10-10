import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.model_selection import train_test_split

from copy import deepcopy


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError

class ProcessMVL1M(DatasetLoader):
    def __init__(self, data_dir, val_ratio = 0.1, test_ratio = 0.1, seed = 0):
        self.fpath_rate = os.path.join(data_dir, 'ratings.dat')
        self.fpath_user = os.path.join(data_dir, 'users.dat')
        self.fpath_item = os.path.join(data_dir, 'movies.dat')
        self.user_list = dict()
        self.user_neg = dict()

        self.train_data = dict()
        self.val_data = dict()
        self.test_data =dict()

        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed= seed

        df_rate = pd.read_csv(self.fpath_rate,
                              sep='::',
                              engine='python',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate'])
        df_user = pd.read_csv(self.fpath_user,
                              sep='::',
                              engine='python',
                              names=['user', 'gender', 'age', 'occupation', 'Zip-code'],
                              usecols=['user', 'gender'])
        df_movies = pd.read_csv(self.fpath_item,
                                 sep='::',
                                 engine='python',
                                 encoding = "ISO-8859-1",
                                 names = ['item','Title','Genres'],
                                 usecols = ['item','Genres'])

        df = pd.merge(df_rate, df_user, on='user')
        df = pd.merge(df,df_movies, on='item')

        # because user and item start from 1, so subtract 1 to get 0
        df['user'] = df['user'].apply(lambda x: x - 1)
        df['item'] = df['item'].apply(lambda x: x - 1)

        df = df.dropna(axis=0, how='any')
        # TODO: save negative rating?
        self.df_negative = df[df['rate'] <=3]
        df = df[df['rate'] > 3]
        self.df = df.reset_index().drop(['index'], axis=1)
    
    def create_user_list(self):
        for row in self.df.itertuples():
            if row.user not in self.user_list:
                self.user_list[row.user] = []
            self.user_list[row.user].append(row.item)
        for row in self.df_negative.itertuples():
            if row.user not in self.user_neg:
                self.user_neg[row.user] = []
            self.user_neg[row.user].append(row.item)

    def split_data(self):

        self.train_data['Positive'] = dict()
        self.val_data['Positive'] = dict()
        self.test_data['Positive'] = dict()

        self.train_data['Negative'] = dict()
        self.val_data['Negative'] = dict()
        self.test_data['Negative'] = dict()
        for user_id, item_list in self.user_list.items():
            if len(item_list) >1:
                train_val_item, test_item = train_test_split(item_list, test_size=self.test_ratio, random_state=self.seed)
                if len(train_val_item) >1:
                    train_item,val_item = train_test_split(train_val_item, test_size=self.val_ratio, random_state=self.seed)
                elif len(train_val_item) == 1:
                    train_item, val_item = train_val_item, train_val_item
            elif len(item_list) == 1:
                train_item = item_list
                test_item = item_list
                val_item = item_list
            self.train_data['Positive'][user_id] = train_item
            self.test_data['Positive'][user_id] = test_item
            self.val_data['Positive'][user_id] = val_item
        for user_id, item_list in self.user_neg.items():
            if len(item_list) >1:
                train_val_item, test_item = train_test_split(item_list, test_size=self.test_ratio, random_state=self.seed)
                if len(train_val_item) >1:
                    train_item,val_item = train_test_split(train_val_item, test_size=self.val_ratio, random_state=self.seed)
                elif len(train_val_item) == 1:
                    train_item, val_item = train_val_item, train_val_item
                # train_item,val_item = train_test_split(train_val_item, test_size=val_ratio, random_state=seed)
            elif len(item_list) == 1:
                train_item = item_list
                test_item = item_list
                val_item = item_list
            self.train_data['Negative'][user_id] = train_item
            self.test_data['Negative'][user_id] = test_item
            self.val_data['Negative'][user_id] = val_item
    def process(self):
        self.create_user_list()
        self.split_data()
        return self.train_data, self.val_data, self.test_data

    