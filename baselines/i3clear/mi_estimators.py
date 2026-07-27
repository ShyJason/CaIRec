# import numpy as np
# import math

# import torch 
# import torch.nn as nn
import torch.nn.functional as F

# class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
#     def __init__(self, x_dim, y_dim, hidden_size):
#         super(CLUBSample, self).__init__()
#         self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
#                                        nn.ReLU(),
#                                        nn.Linear(hidden_size, y_dim))

#         self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
#                                        nn.ReLU(),
#                                        nn.Linear(hidden_size, y_dim),
#                                        nn.Tanh())
        
#         # 对 mu 网络的最后一层初始化
#         nn.init.normal_(self.p_mu[-1].weight, mean=0, std=0.01)
#         nn.init.constant_(self.p_mu[-1].bias, 0)

#         # 对 logvar 网络新增的线性层初始化
#         nn.init.constant_(self.p_logvar[-2].weight, 1.0)
#         nn.init.constant_(self.p_logvar[-2].bias, 0.0)

#     def get_mu_logvar(self, x_samples):
#         mu = self.p_mu(x_samples)
#         logvar = self.p_logvar(x_samples)
#         return mu, logvar
     
        
#     def loglikeli(self, x_samples, y_samples):
#         mu, logvar = self.get_mu_logvar(x_samples)
#         return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
#     def normalization(self, input):
        
#         input_mean = input.mean(dim=0, keepdim=True)
#         input_std = input.std(dim=0, keepdim=True) + 1e-6
#         input_normalized = (input - input_mean) / input_std

#         return input_normalized

#     def forward(self, x_samples, y_samples):
#         # x_samples = self.normalization(x_samples)
#         # y_samples = self.normalization(y_samples)
        
#         # x_samples = F.sigmoid(x_samples)
#         # y_samples = F.sigmoid(y_samples)

#         x_samples = F.normalize(x_samples)
#         y_samples = F.normalize(y_samples)

#         mu, logvar = self.get_mu_logvar(x_samples)
        
#         sample_size = x_samples.shape[0]
#         #random_index = torch.randint(sample_size, (sample_size,)).long()
#         random_index = torch.randperm(sample_size).long()
        
#         positive = - (mu - y_samples)**2 / logvar.exp()
#         negative = - (mu - y_samples[random_index])**2 / logvar.exp()
#         upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
#         return upper_bound/2.

#     def learning_loss(self, x_samples, y_samples):
#         # x_samples = self.normalization(x_samples)
#         # y_samples = self.normalization(y_samples)

#         # x_samples = F.sigmoid(x_samples)
#         # y_samples = F.sigmoid(y_samples)

#         x_samples = F.normalize(x_samples)
#         y_samples = F.normalize(y_samples)

#         return - self.loglikeli(x_samples, y_samples)


import numpy as np
import math
import torch 
import torch.nn as nn
    
class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        x_samples = F.sigmoid(x_samples)
        y_samples = F.sigmoid(y_samples)
        
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
                
        x_samples = F.sigmoid(x_samples)
        y_samples = F.sigmoid(y_samples)
        return - self.loglikeli(x_samples, y_samples)

