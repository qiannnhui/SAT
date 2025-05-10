# Author: Arash Khoeini
# Email: arashkhoeini[at]gmail[dot]com

import torch
import torch.nn.functional as F

class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        """
        Computes the InfoNCE loss.
        
        Args:
            features (torch.Tensor): The feature matrix of shape [2 * batch_size, feature_dim], 
                                     where features[:batch_size] are the representations of 
                                     the first set of augmented images, and features[batch_size:] 
                                     are the representations of the second set.
        
        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        x = features1
        x_aug = features2

        T = self.temperature
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        # features = torch.stack([features1, features2], dim=0)
        # # Normalize features to have unit norm
        # features = F.normalize(features, dim=1)
        
        # # Compute similarity matrix
        # similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # # Get batch size
        # batch_size = features.shape[0] // 2
        
        # # Construct labels where each sample's positive pair is in the other view
        # labels = torch.arange(batch_size, device=features.device)
        # labels = torch.cat([labels + batch_size, labels], dim=0)

        # # Mask out self-similarities by setting the diagonal elements to -inf
        # mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        # similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # # InfoNCE loss
        # loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class SupervisedInfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07, supervised=False):
        """
        kappa_init: Float, = 1 / temperature
        n_samples: int, how many samples to draw from the vMF distributions
        supervised: bool, whether to define positivity/negativity from class labels (target) or to ignore them
                    and only consider the two crops of the same image as positive
        """
        super().__init__()

        self.temperature = temperature
        self.supervised = supervised

    def forward(self, features, target):

        features = F.normalize(features, dim=-1)

        # Calculate similarities
        sim = features.matmul(features.transpose(-2, -1)) / self.temperature

        # Build positive and negative masks
        mask = (target.unsqueeze(1) == target.t().unsqueeze(0)).float()
        pos_mask = mask - torch.diag(torch.ones(mask.shape[0], device=mask.device))
        neg_mask = 1 - mask
        
        # Things with mask = 0 should be ignored in the sum.
        # If we just gave a zero, it would be log sum exp(0) != 0
        # So we need to give them a small value, with log sum exp(-1000) \approx 0
        pos_mask_add = neg_mask * (-1000)
        neg_mask_add = pos_mask * (-1000)

        # calculate the standard log contrastive loss for each vmf sample ([batch])
        log_infonce_per_example = (sim * pos_mask + pos_mask_add).logsumexp(-1) - (sim * neg_mask + neg_mask_add).logsumexp(-1)

        # Calculate loss ([1])
        log_infonce = torch.mean(log_infonce_per_example)
        return -log_infonce