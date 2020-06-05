import numpy as np

from torch.utils.data.sampler import Sampler


class DomainScheduledSampler(Sampler):
    '''
    Sampling according to the score
    '''
    def __init__(self, dataset, sort_idx, ratio, init_ratio, strategy, seed=0):
        np.random.seed(seed)
        if strategy == 'expand':
            start = 0
        elif strategy == 'shift':
            start = int(len(dataset) * (ratio - init_ratio))
        else:
            raise Exception("No such schedule strategy: {}".format(strategy))
        end = int(len(dataset) * ratio)
        self.selected = sort_idx[start:end]
        np.random.shuffle(self.selected)
    
    def __iter__(self):
        return iter(self.selected)

    def __len__(self):
        return len(self.selected)
