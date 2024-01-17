import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from copy import deepcopy


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGeneratorPerUser(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings=None, item_pool=None, num_negatives=None, train_ratio=None, seed=None, column='timestamp'):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.item_pool = item_pool
        self.seed = seed
        self.num_negatives = num_negatives
        self.ratings = ratings
        self.train_ratio = train_ratio
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.train_ratings, self.test_ratings = self._split_train_test(self.preprocess_ratings,
                                                                       self.train_ratio,
                                                                       column)  # self._split_loo(self.preprocess_ratings)
        #print(self.train_ratings)
        #print(self.test_ratings)
        # create negative item samples for NCF learning
        self.train_negatives = self._sample_negative(self.num_negatives)
        #print(self.train_negatives)
        self.is_train = False #self._is_train()

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings_bn = deepcopy(ratings)
        ratings_bn.loc[ratings_bn['rating'] > 0, 'rating'] = 1.0
        return ratings_bn

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _split_train_test(self, ratings, train_ratio, column='timestamp'):
        """leave one out train/test split """
        ratings['ranks'] = ratings.groupby('userId')[column].rank(method='first')
        ratings['counts'] = ratings['userId'].map(ratings.groupby('userId')[column].apply(len))
        ratings['ratio'] = ratings['ranks'] / ratings['counts']
        test = ratings[ratings['ratio'] > train_ratio]
        train = ratings[ratings['ratio'] <= train_ratio]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, num_negatives):
        """return all negative items & 100 sampled negative items"""
        item_pool = self.item_pool - set(self.train_ratings.itemId).union(self.test_ratings.itemId)
        train_negatives = random.sample(list(item_pool), num_negatives)
        return train_negatives

    def _is_train(self):
        """return if sample is used in training"""
        return self.is_train #np.random.uniform(size=1) < 0.1

    def set_is_train(self, is_train):
        """return if sample is used in training"""
        self.is_train = is_train #np.random.uniform(size=1) < 0.1

    def instance_a_train_loader(self):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        for row in self.train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))

        for i in range(len(self.train_negatives)):
           users.append(int(row.userId))
           items.append(int(self.train_negatives[i]))
           ratings.append(float(0))  # negative samples get 0 rating

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))

        return dataset

    def evaluate_data(self):
        """create evaluate data"""
        test_users, test_items, negative_users, negative_items = [], [], [], []
        items = self.item_pool - set(self.train_ratings['itemId']).union(set(self.train_negatives))
        userId = int(pd.unique(self.test_ratings.userId))
        for item in items:
            test_users.append(userId)
            test_items.append(item)
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

