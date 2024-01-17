import pandas as pd
import numpy as np
import math
import scipy.stats
from copy import deepcopy


def compute_precision(tp=None, fp=None):
    return 0.0 if float(tp + fp) == 0.0 else float(tp) / float(tp + fp)


def compute_recall(tp=None, fn=None):
    return 0.0 if float(tp + fn) == 0.0 else float(tp) / float(tp + fn)


def compute_f1(tp=None, fp=None, fn=None):
    return 0.0 if float(2 * tp + fp + fn) == 0.0 else float(2 * tp) / float(2 * tp + fp + fn)


def compute_accuracy(tp=None, tn=None, fp=None, fn=None):
    return 0.0 if float(tp + fp + fn + tn) == 0.0 else float(tp + tn) / float(tp + fp + fn + tn)


def compute_mean_precision_k(actual, predicted, k):
    if len(predicted) > k:
        predicted_temp = predicted[:k]
    else:
        predicted_temp = predicted

    score = 0.0
    num_hit = 0.0

    for i, p in enumerate(predicted_temp):
        if p in actual and p not in predicted_temp[:i]:
            num_hit += 1.0
            score += num_hit / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


class MetricsAtTopK(object):
    def __init__(self, top_k=None, training_items=None, observed_items=None):
        self.top_k = top_k
        self.observed_items = observed_items
        self.training_items = training_items
        self.predicted_ratings = None  # Subjects which we ran evaluation on
        self.predicted_items = None
        self.top_k_predicted_ratings = None
        self.top_k_predicted_items = None

    def get_top_k(self):
        return self.top_k

    def set_top_k(self, top_k):
        self.top_k = top_k

    def get_observed_items(self):
        return self.observed_items

    def set_observed_items(self, observed_items):
        self.observed_items = observed_items

    def get_training_items(self):
        return self.training_items

    def set_training_items(self, training_items):
        self.training_items = training_items

    def get_predicted_ratings(self):
        return self.predicted_ratings

    def set_predicted_ratings_top_list(self, test_users, top_list_dict):
        predicted_ratings_df = pd.DataFrame.from_dict(top_list_dict, orient='index', columns=['score'])
        predicted_ratings_df['user'] = np.repeat(test_users.unique(), len(predicted_ratings_df))
        predicted_ratings_df['rank'] = predicted_ratings_df['score'].rank(method='first', ascending=False)
        predicted_ratings_df.sort_values(['rank'], inplace=True)
        self.predicted_ratings = predicted_ratings_df
        del predicted_ratings_df

    def set_predicted_ratings(self, predicted_ratings):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(predicted_ratings, list)
        test_users, test_items, test_scores = predicted_ratings[0], predicted_ratings[1], predicted_ratings[2]
        # the ground-truth set
        predicted_ratings_df = pd.DataFrame({'user': test_users,
                                             'test_item': test_items,
                                             'score': test_scores})
        predicted_ratings_df['rank'] = predicted_ratings_df['score'].rank(method='first', ascending=False)
        predicted_ratings_df.sort_values(['user', 'rank'], inplace=True)
        self.predicted_ratings = predicted_ratings_df
        self.predicted_ratings.set_index('test_item', inplace=True)
        del predicted_ratings_df

    def get_predicted_items(self):
        return self.predicted_items

    def infer_predicted_items(self):
        self.predicted_items = list(self.predicted_ratings.index)

    def get_top_k_predicted_ratings(self):
        return self.top_k_predicted_ratings

    def infer_top_k_predicted_ratings(self):
        top_k = self.predicted_ratings[self.predicted_ratings['rank'] <= self.top_k]
        self.top_k_predicted_ratings = top_k
        del top_k

    def get_top_k_predicted_items(self):
        return self.top_k_predicted_items

    def infer_top_k_predicted_items(self):
        self.top_k_predicted_items = list(self.top_k_predicted_ratings.index)
        if len(self.top_k_predicted_items) != 10:
            print(self.top_k_predicted_items)

    def exclude_training_samples(self):
        if self.training_items is not None:
            for val in self.training_items:
                if val in self.observed_items:
                    self.observed_items.remove(val)
                if val in self.predicted_items:
                    self.predicted_items.remove(val)
                    self.predicted_ratings = self.predicted_ratings.drop(val)

    def compute_metrics(self):

        tp = 0.0
        fp = 0.0
        fn = 0.0
        tn = 0.0
        hr = 0.0

        for val in self.top_k_predicted_items:
            if val in self.observed_items:
                tp += 1.0
            else:
                fp += 1.0

        for val in self.observed_items:
            if val not in self.top_k_predicted_items:
                fn += 1.0

        for val in self.predicted_items[self.top_k:]:
            if val not in self.observed_items:
                tn += 1.0

        for val in self.observed_items:
            if val in self.top_k_predicted_items:
                hr += 1

        m_a_p = self.compute_map()

        return tp, fp, fn, tn, hr, m_a_p

    def compute_ndcg(self):
        test_in_top_k = deepcopy(self.top_k_predicted_ratings)
        test_in_top_k = test_in_top_k.loc[[x for x in self.observed_items if x in list(test_in_top_k.index)]]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        ndcg = test_in_top_k['ndcg'].sum() * 1.0 / self.predicted_ratings['user'].nunique()
        del test_in_top_k
        return ndcg

    def compute_map(self):
        return np.mean([compute_mean_precision_k(self.observed_items, self.predicted_items, self.top_k) for a, p in
                        zip(self.observed_items, self.predicted_items)])

    def compute_nmr(self):

        nmr = 1
        mrr = 0
        # actual : 1-D np.array (indices of items which are to be predicted)
        # train : list (similar to actual but training set)
        # predicted: list (recommendation scores for all the items. Position represent the item index [local indexing])

        float_max = np.finfo(np.float32).max
        predicted = -np.asarray(deepcopy(self.predicted_ratings.score))

        # if self.training_items is not None:
        #    predicted[self.training_items] = np.finfo(np.float32).max

        ranked_items = scipy.stats.rankdata(predicted, method='average')
        # print(ranked_items)
        # print(len(ranked_items))

        ranked_items = pd.DataFrame(ranked_items, index=self.predicted_ratings.index, columns=['rank'])
        #print(ranked_items)
        #print(self.observed_items)
        index = [x for x in self.observed_items if x in ranked_items.index]
        if index:
            ranked_items = ranked_items.loc[index]

            ranked_items = np.array(ranked_items['rank'])
            mrr = np.mean(1 / ranked_items)
            nmr = np.mean(ranked_items) / len(predicted)

        del predicted

        return nmr, mrr

    def compute_nmr_top_list(self):

        # actual : 1-D np.array (indices of items which are to be predicted)
        # train : list (similar to actual but training set)
        # predicted: list (recommendation scores for all the items. Position represent the item index [local indexing])

        float_max = np.finfo(np.float32).max
        predicted = -np.asarray(deepcopy(self.predicted_ratings.score))

        # if self.training_items is not None:
        #    predicted[self.training_items] = np.finfo(np.float32).max

        ranked_items = scipy.stats.rankdata(predicted, method='average')
        # print(ranked_items)
        # print(len(ranked_items))

        ranked_items = pd.DataFrame(ranked_items, index=self.predicted_ratings.index, columns=['rank'])

        index = [x for x in self.observed_items if x in ranked_items.index]

        ranked_items = ranked_items.loc[index]

        ranked_items = np.array(ranked_items['rank'])
        mrr = np.mean(1 / ranked_items)
        nmr = np.mean(ranked_items) / len(predicted)

        del predicted

        return nmr, mrr