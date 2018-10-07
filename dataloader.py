import torch
import torch.utils.data as torch_data
import os
import data.utils

class dataset(torch_data.Dataset):

    def __init__(self, src, tgt, raw_src, raw_tgt):

        self.src = src
        self.tgt = tgt
        self.raw_src = raw_src
        self.raw_tgt = raw_tgt

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], self.raw_src[index], self.raw_tgt[index]

    def __len__(self):
        return len(self.src)


def load_dataset(path):
    pass


def save_dataset(dataset, path):
    if not os.path.exists(path):
        os.mkdir(path)


def padding(data):
    src, tgt, raw_src, raw_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = s[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = s[:end]

    rvr_tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        s_list = s[:end].tolist()
        rvr_tgt_pad[i, :end] = torch.LongTensor(s_list[:1] + s_list[1:-1][::-1] + s_list[-1:])

    return raw_src, src_pad.t(), torch.LongTensor(src_len), \
           raw_tgt, tgt_pad.t(), rvr_tgt_pad.t(), torch.LongTensor(tgt_len)


def get_loader(dataset, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=padding)
    return data_loader