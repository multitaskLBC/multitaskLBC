import torch


def confusion_weight(n_categories_total, subset, device = 'cuda'):
    '''
    helper function to correct for imbalance in number of samples
    per category

    n_categories_total: total number of categories that is sampled from
    subset: a list of the indexes for which to calculate the loss
    '''
    return torch.arange(1, n_categories_total, device = device)[subset] / n_categories_total, device = device)


class LBCWithLogitsLoss:
    def __init__(self, n_categories_total, subset, device = 'cuda'):
        '''
        the custom loss function for unbiased, multi-task learning by confusion (LBC)

        n_categories_total: total number of categories that is sampled from
        subset: a list of the indexes for which to calculate the loss
        '''
        self.pos_weight = confusion_weight(n_categories_total, subset, device)
        self.logsig = torch.nn.LogSigmoid()
        
    def __call__(self, logits, targets):
        loss = - (targets * self.logsig(logits) / (1. - self.pos_weight)
                 +(1. - targets) * self.logsig(-logits) / (self.pos_weight)
                 )   
        return torch.mean(loss)

    
def lbc_label(y, subset, device = 'cuda'):
    '''
    turn label number into label vector

    y: labels of the samples as numbers, e.g. 0 for the first category.
    subset: a list of the indexes for which to calculate the loss.
    '''
    return torch.tensor(subset, device = device).view(1,-1) < y.view(-1,1)
