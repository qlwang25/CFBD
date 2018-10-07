import torch
import torch.nn as nn
import models
import data.dict as dict
from torch.autograd import Variable


def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit

# def cross_entropy_loss(hidden_outputs, targets, decoder, criterion, config):
#     loss = torch.FloatTensor([0])
#     if config.use_cuda:
#         loss = loss.cuda()

#     for i, penalty in zip(range(3), [1, 1, 1.5]):
#         outputs = hidden_outputs[i].view(-1, hidden_outputs[i].size(2))
#         scores = decoder.score_fn(outputs)

#         loss += criterion(scores, targets[i].view(-1)) * penalty

#     pred = scores.max(1)[1]
#     num_correct = pred.eq(targets[-1].view(-1)).masked_select(targets[-1].view(-1).ne(dict.PAD)).sum()
#     num_total = targets[-1].ne(dict.PAD).sum()
    
#     loss.div(num_total.float()).backward()

#     return loss, num_total, num_correct

