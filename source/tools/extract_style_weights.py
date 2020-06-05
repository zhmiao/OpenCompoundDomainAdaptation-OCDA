import collections
import torch

def extract_style_weights(infile, outfile, header):
    weights = torch.load(infile)
    new_weights = collections.OrderedDict()
    for k in weights.keys():
        if k.startswith(header):
            new_weights[k[len(header) + 1:]] = weights[k]
    torch.save(new_weights, outfile)
