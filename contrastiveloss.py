import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def contrastive_unlabeled(src, trg, lbl, temperature, subset_size=2000):
    src = F.normalize(src, dim=1)
    trg = F.normalize(trg, dim=1)
    assert src.dim() == 4
    assert trg.dim() == 4
    assert src.size(0) == trg.size(0), "{0} vs {1} ".format(src.size(0), trg.size(0))
    assert src.size(1) == trg.size(1), "{0} vs {1} ".format(src.size(2), trg.size(1))
    assert src.size(2) == trg.size(2), "{0} vs {1} ".format(src.size(3), trg.size(3))
    assert src.size(3) == trg.size(3), "{0} vs {1} ".format(src.size(0), trg.size(0))

    n, c, h, w = src.size()
    lbl_mask = (lbl >= 0) * (lbl != 255)
    lbl = lbl[lbl_mask]
    
    if not lbl.data.dim():
        return Variable(torch.zeros(1))
    # cleaning 0 and 255 label pixels 
    src = src.transpose(1, 2).transpose(2, 3).contiguous()
    src = src[lbl_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

    trg = trg.transpose(1, 2).transpose(2, 3).contiguous()
    trg = trg[lbl_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    
    lbl = lbl.contiguous().view(-1, 1)

    # selecting a subset to do contrastive learning
    pix, _ = src.size()
    subset = torch.randint(0,pix,(subset_size,)).long()
    
    src = src[subset]
    trg = trg[subset]
    lbl = lbl[subset]
 
    src_scores, src_preds = torch.max(src, dim=1, keepdim=True)
    trg_scores, trg_preds = torch.max(trg, dim=1, keepdim=True)
    mask = torch.eye(subset_size,dtype=torch.uint8).cuda()


    logit_mask = torch.eq(src_preds,trg_preds.transpose(0,1)) | mask.bool() # logit_mask[i,j]=1 if i,j is positive pair when i\neq j 
    similarity = torch.matmul(src,trg.transpose(0,1)) / temperature
    # for numerical stability
    logits_max, _ = torch.max(similarity, dim=1)
    logits = similarity - logits_max.detach()
    negatives = (1 - logit_mask.float())*torch.exp(logits)

    # log prob = pos- torch.log(pos(i,i) + rest_of_negatives)
    log_prob = logits - torch.log(mask.float()*torch.exp(logits) + negatives.sum(1, keepdim=True))
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask.float() * log_prob).sum(1)

    # loss
    loss = - mean_log_prob_pos
    loss = loss.mean()
     
    return loss

def contrastive_labeled(src, trg, lbl, temperature, subset_size=2000):
    src = F.normalize(src, dim=1)
    trg = F.normalize(trg, dim=1)
    assert not lbl.requires_grad
    assert src.dim() == 4
    assert trg.dim() == 4
    assert lbl.dim() == 3
    assert src.size(0) == lbl.size(0), "{0} vs {1} ".format(src.size(0), lbl.size(0))
    assert src.size(2) == lbl.size(1), "{0} vs {1} ".format(src.size(2), lbl.size(1))
    assert src.size(3) == lbl.size(2), "{0} vs {1} ".format(src.size(3), lbl.size(3))
    assert trg.size(0) == lbl.size(0), "{0} vs {1} ".format(trg.size(0), lbl.size(0))
    assert trg.size(2) == lbl.size(1), "{0} vs {1} ".format(trg.size(2), lbl.size(1))
    assert trg.size(3) == lbl.size(2), "{0} vs {1} ".format(trg.size(3), lbl.size(3))
    n, c, h, w = src.size()
    lbl_mask = (lbl >= 0) * (lbl != 255)
    lbl = lbl[lbl_mask]
    if not lbl.data.dim():
        return Variable(torch.zeros(1))
    
    src = src.transpose(1, 2).transpose(2, 3).contiguous()
    src = src[lbl_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    
    trg = trg.transpose(1, 2).transpose(2, 3).contiguous()
    trg = trg[lbl_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    
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

def contrastive_labeled_new_sampling(src, trg, lbl, temperature, subset_size=2000):
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
    
    # sampling = lbl_line == (0 or 1 or 
                            
    
    
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

def contrastive_labeled_new_debug(src, trg, src_pred, lbl, temperature, subset_size=2000):
    src = F.normalize(src, dim=1)
    trg = F.normalize(trg, dim=1)
    n, c, h, w = src.size()
    
    src = src.transpose(1, 2).transpose(2, 3).contiguous()
    src = src.view(-1, c)
    # size of source is (n*h*w,c)
    
    trg = trg.transpose(1, 2).transpose(2, 3).contiguous()
    trg = trg.view(-1, c)
    
    # labels from prediction of model
    lbl_pred = torch.argmax(src_pred, dim=1)
    
    lbl_line = lbl.contiguous().view(-1, 1)
    lbl_pred_line = lbl_pred.contiguous().view(-1, 1)
    
    pix, _ = src.size()
    subset = torch.randint(0,pix,(subset_size,)).long()
    
    src = src[subset]
    trg = trg[subset]
    lbl_line = lbl_line[subset]
    lbl_pred_line = lbl_pred_line[subset]
    
    # mask from ground truth labels
    mask = torch.eq(lbl_line,lbl_line.transpose(0,1))
    
    # mask from predictions
    mask_pred = torch.eq(lbl_pred_line,lbl_pred_line.transpose(0,1))
    
    false_neg = torch.logical_and(mask==True, mask_pred==False).sum()
    false_pos = torch.logical_and(mask==False, mask_pred==True).sum()
    
    ## calculation of contrastive loss with ground truth labels
    similarity = torch.matmul(src,trg.transpose(0,1)) / temperature
    # for numerical stability
    logits_max, _ = torch.max(similarity, dim=1)
    logits = similarity - logits_max.detach()
    negatives = (1 - mask.float())*torch.exp(logits)
    # log prob = pos- torch.log(pos+negatives)
    log_prob = logits - torch.log(mask.float()*torch.exp(logits) + negatives.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask.float() * log_prob).sum(1) / mask.sum(1).float()
    loss_truth = - mean_log_prob_pos
    loss_truth = loss_truth.mean()
    
    ## calculation of contrastive loss with predicted labels
    similarity_pred = torch.matmul(src,trg.transpose(0,1)) / temperature
    # for numerical stability
    logits_max, _ = torch.max(similarity_pred, dim=1)
    logits = similarity_pred - logits_max.detach()
    negatives = (1 - mask_pred.float())*torch.exp(logits)
    # log prob = pos- torch.log(pos+negatives)
    log_prob = logits - torch.log(mask_pred.float()*torch.exp(logits) + negatives.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_pred.float() * log_prob).sum(1) / mask_pred.sum(1).float()
    loss_pred = - mean_log_prob_pos
    loss_pred = loss_pred.mean()
    return loss_truth, loss_pred, false_neg, false_pos


def forward(self, features, labels=None, mask=None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
              has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
    """
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if self.contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif self.contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
          torch.ones_like(mask),
          1,
          torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
          0
    )
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    # loss
    loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss

if __name__ == "__main__":
    torch.manual_seed(0)
    src=torch.rand(1,19,512,1024)
    trg=torch.rand(1,19,512,1024)
    lbl=torch.randint(1,19,(1,512,1024))
    loss = contrastive_labeled(src, trg, lbl ,1)
    # print(loss)
