import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def clip_loss(logits_per_smile, logits_per_text):
    device = logits_per_smile.device
    B = logits_per_smile.shape[0]
    labels = torch.arange(logits_per_smile.shape[0], device=device, dtype=torch.long)
    CL_loss = (
        F.cross_entropy(logits_per_smile, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2.0
    
    pred_smile = logits_per_smile.argmax(dim=1, keepdim=False)
    pred_text = logits_per_text.argmax(dim=1, keepdim=False)
    CL_acc = (pred_smile.eq(labels).sum().detach().cpu().item() * 1. / B +
              pred_text.eq(labels).sum().detach().cpu().item() * 1. / B
    ) / 2.0

    return { "CL_loss": CL_loss, "CL_acc": CL_acc }

class CrossEn(nn.Module):
    def __init__(self):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

def STM_CL(logits_per_smile, logits_per_text):

    criterion = nn.CrossEntropyLoss()
    B = logits_per_smile.size()[0]
    labels = torch.arange(B).long().to(logits_per_smile.device)  # B*1

    CL_loss = (criterion(logits_per_smile, labels) +
               criterion(logits_per_text, labels) 
    ) / 2.0
    
    pred_smile = logits_per_smile.argmax(dim=1, keepdim=False)
    pred_text = logits_per_text.argmax(dim=1, keepdim=False)
    CL_acc = (pred_smile.eq(labels).sum().detach().cpu().item() * 1. / B +
              pred_text.eq(labels).sum().detach().cpu().item() * 1. / B
    ) / 2.0

    return { "CL_loss": CL_loss, "CL_acc": CL_acc }