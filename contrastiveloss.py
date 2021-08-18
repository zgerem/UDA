import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def contrastive_labeled_new(src, trg, lbl, temperature, subset_size=2000):
    src = F.normalize(src, dim=1)
    trg = F.normalize(trg, dim=1)
    n, c, h, w = src.size()
    
    src = src.transpose(1, 2).transpose(2, 3).contiguous()
    src = src.view(-1, c)
    # size of source is (n*h*w,c)
    
    trg = trg.transpose(1, 2).transpose(2, 3).contiguous()
    trg = trg.view(-1, c)
    
    lbl_line = lbl.contiguous().view(-1, 1)
    pix, _ = src.size()
    subset = torch.randint(0,pix,(subset_size,)).long()
    
    
    
    
    src = src[subset]
    trg = trg[subset]
    lbl_line = lbl_line[subset]
    
   
    
    mask = torch.eq(lbl_line,lbl_line.transpose(0,1))
    similarity = torch.matmul(src,trg.transpose(0,1)) / temperature
    # for numerical stability
    logits_max, _ = torch.max(similarity, dim=1)
    logits = similarity - logits_max.detach()
    negatives = (1 - mask.float())*torch.exp(logits)
    # log prob = pos- torch.log(pos+negatives)
    log_prob = logits - torch.log(mask.float()*torch.exp(logits) + negatives.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask.float() * log_prob).sum(1) / mask.sum(1).float()
    loss = - mean_log_prob_pos
    loss = loss.mean()
    return loss

