import torch
from collections import OrderedDict


class FedAvg():
    def __init__(self, param_keys, client_model_names, lr=None):
        self.param_keys = param_keys # does not include params with requires_grad == False
        self.client_model_names = client_model_names

    def update(self, orig_model_weights, client_model_weights, client_weights):
        # orig_model_weights: global model weights
        # client_weights: number of data samples
        updated_model_weights = {}
        for k, p in orig_model_weights.items():
            new_p = self.update_k(k, p, client_model_weights, client_weights)
            if new_p is None:
                new_p = p
            updated_model_weights[k] = new_p

        return updated_model_weights

    def update_k(self, k, orig_p, new_model_weights, client_weights):
        k_update_weights = []
        stacked_model_weights = []
        for c, model_weights in new_model_weights.items():
            if k in model_weights and orig_p.size() == model_weights[k].size():
                k_update_weights.append(client_weights[c])
                stacked_model_weights.append(model_weights[k].cpu())
            elif k.startswith(self.client_model_names[c]):
                model_k = k[len(self.client_model_names[c]) + 1:]
                if model_k in model_weights and orig_p.size() == model_weights[model_k].size():
                    k_update_weights.append(client_weights[c])
                    stacked_model_weights.append(model_weights[model_k].cpu())

        if not len(k_update_weights):
            return

        k_update_weights = torch.FloatTensor(k_update_weights)
        k_update_weights /= k_update_weights.sum()
        #print('k_update_weights', k, k_update_weights)
        for _ in orig_p.shape:
            k_update_weights = k_update_weights.unsqueeze(-1)

        return (k_update_weights * torch.stack(stacked_model_weights, 0)).sum(0)


class FedAdam(FedAvg):
    def __init__(self, param_keys, client_model_names, lr=10.):
        super().__init__(param_keys, client_model_names)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1.
        self.lr = lr
        self.m, self.v = OrderedDict(), OrderedDict()
        for k in self.param_keys:
            self.m[k], self.v[k] = 0., 0.

    def update(self, orig_model_weights, client_model_weights, client_weights):
        updated_model_weights = {}
        for k, p in orig_model_weights.items():
            new_weight = self.update_k(k, p, client_model_weights, client_weights)
            if new_weight is None:
                continue
            delta_weight = new_weight - orig_model_weights[k]

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * delta_weight
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * torch.square(delta_weight)
            updated_model_weights[k] = p + self.lr * self.m[k] / (torch.sqrt(self.v[k]) + self.epsilon)

        return updated_model_weights