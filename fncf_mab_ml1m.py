import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math
from copy import deepcopy
import pickle
import gc
from bayesian_thompson_sampling import GaussianThompsonItem
from gradient_optimizer import GradientOptimzer
from collections import OrderedDict
import copy

random.seed(0)

def index_argmax(value_list, topN=10):
    """ return the index of topN max elements"""
    values = np.asarray(value_list)
    return np.argsort(values)[::-1][:topN]

def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # the full set
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores})
        full = pd.merge(full, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()


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


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for NCF learning
        self.negatives = self._sample_negative(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class FLClient(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.clientModel = NeuMF(config)
        self.clientModel_Opt = use_optimizer(self.clientModel, config)

        # exdeepcopycit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'clientModel'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        self.clientModel_Opt.zero_grad()
        ratings_pred = self.clientModel(users, items)
        ratings.require_grad = False
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        # self.opt.step()
        loss = loss.item()
        return loss

    def get_local_model_updates(self):
        with torch.no_grad():
            grad_dict = {k: v.grad for k, v in self.clientModel.named_parameters()}
        return grad_dict

    def set_local_model_weights(self, weights, selected_items_index=None):
        self.clientModel_Opt.zero_grad()
        with torch.no_grad():
            for k, v in self.clientModel.named_parameters():
                if k == "embedding_item_mf.weight":
                    v.data[selected_items_index] = weights[k]
                else:
                    v.data = weights[k]
        del weights


class FederatedRecommendationModel(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self.serverModel = NeuMF(config)
        self.serverModel_Opt = use_optimizer(self.serverModel, config)
        self.serverModel.train()
        self.mabModel = [GaussianThompsonItem() for i in range(self.config['num_items'])]
        self.gradientOptimzer = GradientOptimzer(self.config['num_items'], self.config['latent_dim_mf'])
        self.global_model_gradients = OrderedDict()
        self.past_gradients = np.zeros((self.config['num_items'], self.config['latent_dim_mf']))
        # exdeepcopycit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def init_global_model_gradients(self):
        with torch.no_grad():
            for k, v in self.serverModel.named_parameters():
                self.global_model_gradients[k] = v.grad

    def get_global_model(self):
        weights = {}
        with torch.no_grad():
            for name, param in self.serverModel.named_parameters():
                weights[name] = param.data
        return weights

    # def update_global_model(self, gradients):
    #    self.opt.zero_grad()
    #    with torch.no_grad():
    #        for k, v in self.serverModel.named_parameters():
    #            v.grad = gradients[k]
    #    self.opt.step()

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'serverModel'), 'Please specify the exact model !'
        total_loss = 0
        model_updates = []
        weights = self.get_global_model()

        item_samples = [item.sample() for item in self.mabModel]

        selected_items_index = index_argmax(item_samples, int(config['num_items'] * self.config['train_user_ratio']))

        selected_items = torch.tensor(copy.deepcopy(selected_items_index))

        selected_payload = OrderedDict()
        full_payload = weights
        for key in full_payload:
            if key == "embedding_item_mf.weight":
                selected_payload[key] = full_payload[key][selected_items]
            else:
                selected_payload[key] = full_payload[key]

        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            clientModel = FLClient(self.config)
            clientModel.set_local_model_weights(selected_payload, selected_items)
            loss = clientModel.train_single_batch(user, item, rating)
            local_model_updates = clientModel.get_local_model_updates()
            model_updates.append(local_model_updates)
            del clientModel
            # print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss

        print('model/loss', total_loss, epoch_id)

        agg_gradients = {}
        for key in model_updates[0]:
            for l in model_updates:
                if key in agg_gradients:
                    agg_gradients[key] += l[key]
                else:
                    agg_gradients[key] = l[key]

        for key in self.global_model_gradients:
            # print(agg_gradients[key].size())
            if key == "embedding_item_mf.weight":
                self.global_model_gradients[key][selected_items] = avg_gradients[key][selected_items]
            else:
                self.global_model_gradients[key] = avg_gradients[key]
        # self.update_global_model(agg_gradients)
        self.serverModel_Opt.zero_grad()
        with torch.no_grad():
            for k, v in self.serverModel.named_parameters():
                v.grad = agg_gradients[k]
        self.serverModel_Opt.step()

        agg_gradients = agg_gradients['embedding_item_mlp.weight'].numpy()
        curr_grad = agg_gradients[selected_items_index]
        hist_gradients = self.gradientOptimzer.historical_gradient_estimates(curr_grad, selected_items_index)
        hist_grad = hist_gradients[selected_items_index]
        prev_grad = self.past_gradients[selected_items_index]
        diff = np.sum(np.abs(prev_grad - curr_grad), axis=1)
        cos_sim = np.sum(np.abs(hist_grad - curr_grad), axis=1)
        reward = (((1 - self.config['alpha']) * epoch) * (cos_sim)) + ((self.config['alpha'] / epoch) * diff)
        max_reward = np.max(reward)
        if max_reward != 0:
            reward = reward / max_reward
        # else:
        #    reward = reward / 1

        item_cnt = 0
        for i in selected_items_index:
            if reward[item_cnt] is not np.nan:
                self.mabModel[i].update(reward[item_cnt])
            else:
                self.mabModel[i].update(0)
            item_cnt = item_cnt + 1

        self.past_gradients[selected_items_index] = curr_grad

        del selected_items_index
        del selected_items
        del agg_gradients
        gc.collect()
        return total_loss

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'serverModel'), 'Please specify the exact model !'

        self.serverModel.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()
            test_scores = self.serverModel(test_users, test_items)
            negative_scores = self.serverModel(negative_users, negative_items)
            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                     test_items.data.view(-1).tolist(),
                                     test_scores.data.view(-1).tolist(),
                                     negative_users.data.view(-1).tolist(),
                                     negative_items.data.view(-1).tolist(),
                                     negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        print('performance/HR', hit_ratio, epoch_id)
        print('performance/NDCG', ndcg, epoch_id)
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg

    # def save(self, alias, epoch_id, hit_ratio, ndcg):
    #    assert hasattr(self, 'model'), 'Please specify the exact model !'
    #    model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
    #    save_checkpoint(self.model, model_dir)


if __name__ == "__main__":
    # Load Data
    ml1m_dir = 'data/original_data/ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                              engine='python')
    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
    # DataLoader for training
    sample_generator = SampleGenerator(ratings=ml1m_rating)
    evaluate_data = sample_generator.evaluate_data
    gc.collect()

    neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                    'num_epoch': 500,
                    'batch_size': 32,
                    'optimizer': 'adam',
                    'adam_lr': 5e-2,
                    'num_users': 6040,
                    'num_items': 3706,
                    'latent_dim_mf': 4,
                    'latent_dim_mlp': 4,
                    'num_negative': 4,
                    'layers': [8, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                    'l2_regularization': 0.0000001,
                    'train_user_ratio': 0.1,
                    'alpha': 0.999,
                    'use_cuda': False,
                    'device_id': 7,
                    'pretrain': False,
                    'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                    'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                    }

    config = neumf_config
    items = [GaussianThompsonItem() for i in range(config['num_items'])]

    fedRecomModel = FederatedRecommendationModel(config)
    fl_results = {'loss': [], 'hit_ratio': [], 'ndcg': []}

    for epoch in range(1, config['num_epoch'] + 1):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        loss = fedRecomModel.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = fedRecomModel.evaluate(evaluate_data, epoch_id=epoch)
        # engine.save(config['alias'], epoch, hit_ratio, ndcg)
        print(hit_ratio, ndcg)
        fl_results['loss'].append(loss)
        fl_results['hit_ratio'].append(hit_ratio)
        fl_results['ndcg'].append(ndcg)
        gc.collect()

    with open("results/ml-1m/updated_results/payload_fed_neucf.pkl", 'wb') as fp:
        pickle.dump(fl_results, fp)

    gc.collect()