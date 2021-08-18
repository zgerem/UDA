import torch
import torch.nn.functional as F

def contrastive_unlabeled_old(src, trg, temperature):
    src = F.normalize(src, dim=1)
    trg = F.normalize(trg, dim=1)
    assert src.dim() == 4
    assert trg.dim() == 4
    assert src.size(0) == trg.size(0), "{0} vs {1} ".format(src.size(0), trg.size(0))
    assert src.size(1) == trg.size(1), "{0} vs {1} ".format(src.size(2), trg.size(1))
    assert src.size(2) == trg.size(2), "{0} vs {1} ".format(src.size(3), trg.size(3))
    assert src.size(3) == trg.size(3), "{0} vs {1} ".format(src.size(0), trg.size(0))

    n, c, h, w = src.size()
    src = src.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    trg = trg.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    similarity = (src*trg).sum(dim=1) / temperature
    # for numerical stability
    logits_max, _ = torch.max(similarity, dim=0)
    logits = similarity - logits_max.detach()
    probs = F.softmax(logits, dim=0)
    log_prob = torch.log(probs)

    # compute mean of log-likelihood over positive
    loss = -log_prob.mean()
    return loss

def contrastive_unlabeled_new(src, trg, temperature):
    src = F.normalize(src, dim=1)
    trg = F.normalize(trg, dim=1)
    assert src.dim() == 4
    assert trg.dim() == 4
    assert src.size(0) == trg.size(0), "{0} vs {1} ".format(src.size(0), trg.size(0))
    assert src.size(1) == trg.size(1), "{0} vs {1} ".format(src.size(2), trg.size(1))
    assert src.size(2) == trg.size(2), "{0} vs {1} ".format(src.size(3), trg.size(3))
    assert src.size(3) == trg.size(3), "{0} vs {1} ".format(src.size(0), trg.size(0))

    n, c, h, w = src.size()
    src = src.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    trg = trg.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    _, src_lbls = torch.max(src, dim=1)
    _, trg_lbls = torch.max(trg, dim=1)
    # indexes=(src_lbls != trg_lbls).nonzero(as_tuple=False) pytorch changed this
    indexes=(src_lbls != trg_lbls).nonzero()[:,0] #older version
    src = src[indexes]
    trg = trg[indexes]
    #print(indexes.shape)
    #print("source:",src.size())
    pix, c = src.size()
    mask = torch.eye(pix)
    similarity = torch.matmul(src,trg.transpose(1,0)) / temperature
    # for numerical stability
    logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
    logits = similarity - logits_max.detach()
    exp_logits = torch.exp(logits) * mask.float()
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask.float() * log_prob).sum(1) / mask.sum(1).float()

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
    print(n,c,h,w)
    lbl_mask = (lbl >= 0) * (lbl != 255)
    lbl = lbl[lbl_mask]
    if not lbl.data.dim():
        return Variable(torch.zeros(1))
    
    subset = torch.randint(0,n*h*w,subset_size)
    src = src.transpose(1, 2).transpose(2, 3).contiguous()
    src = src[lbl_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    src = src[subset]
    
    trg = trg.transpose(1, 2).transpose(2, 3).contiguous()
    trg = trg[lbl_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    trg = trg[subset]
    
    lbl_line = lbl.contiguous().view(-1, 1)
    lbl_line = lbl_line[subset]
    mask = torch.eq(lbl_line,lbl_line.transpose(1,0))
    similarity = torch.matmul(src,trg.transpose(1,0)) / temperature
    # for numerical stability
    logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
    logits = similarity - logits_max.detach()
    exp_logits = torch.exp(logits) * mask.float()
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask.float() * log_prob).sum(1) / mask.sum(1).float()

    # loss
    loss = - mean_log_prob_pos
    loss = loss.mean()
    return loss

if __name__ == "__main__":
    torch.manual_seed(0)
    src=torch.rand(1,19,512,1024)
    trg=torch.rand(1,19,512,1024)
    lbl=torch.randint(1,19,(1,512,1024))
    loss = contrastive_labeled(src, trg, lbl ,1)
    print(loss)
