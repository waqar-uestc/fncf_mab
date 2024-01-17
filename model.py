import torch
from utils import save_checkpoint, use_optimizer
from metrics import MetricsAtTopK


class RecommendationModel(object):
    """Meta Class for training & evaluating Recommendation model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss_value = loss.item()
        loss.backward()
        self.opt.step()
        return loss_value

    def train_an_epoch(self, train_loader):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        assert isinstance(train_loader.user_tensor, torch.LongTensor)
        #self.model.train()
        #print(train_loader.user_tensor)
        #print(train_loader.item_tensor)
        #print(train_loader.target_tensor)
        model_loss = self.train_single_batch(train_loader.user_tensor, train_loader.item_tensor, train_loader.target_tensor.float())
        return model_loss

    def evaluate(self, evaluate_data=None, top_k=None, training_items=None, observed_items=None):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        with torch.no_grad():
            test_scores = self.model(test_users, test_items)

        metron = MetricsAtTopK(top_k=top_k, training_items=training_items, observed_items=observed_items)
        metron.set_predicted_ratings([test_users.data.view(-1).tolist(),
                             test_items.data.view(-1).tolist(),
                             test_scores.data.view(-1).tolist()])
        metron.infer_predicted_items()
        #metron.exclude_training_samples()
        metron.infer_top_k_predicted_ratings()
        metron.infer_top_k_predicted_items()
        tp, fp, fn, tn, hit_ratio, mp = metron.compute_metrics()
        ndcg =  metron.compute_ndcg()
        nmr, mrr = metron.compute_nmr()
        return hit_ratio, ndcg, mp, nmr, mrr, tp, fp, fn, tn

    def save(self, alias, dataset, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, dataset, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)