import pandas as pd
from torch.utils.data import Dataset, DataLoader

import sys

class UtteranceDataset(Dataset):

    # content是某对话的[utterance1, utterance2, ...]
    # utterances是[conversation1, conversation2, ...]
    def __init__(self, filename1, filename2, filename5, filename6):

        # print(filename1)
        # datasets/dialogue_level_minibatch/meld/train_prompted_utterances.txt
        utterances, labels, loss_mask, speakers = [], [], [], []

        
        with open(filename1, 'r', encoding='utf-8') as f:
            for line in f:
                content = line.strip()
                utterances.append(content) 

        with open(filename2, 'r', encoding='utf-8') as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                labels.extend([int(l) for l in content])

        with open(filename5, 'r', encoding='utf-8') as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                loss_mask.extend([int(l) for l in content])

        with open(filename6, 'r', encoding='utf-8') as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                speakers.extend(content)

        self.utterances = utterances
        self.labels = labels
        self.loss_mask = loss_mask
        self.speakers = speakers
        
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index): 
        s = self.utterances[index]
        l = self.labels[index]
        m = self.loss_mask[index]
        sp = self.speakers[index]
        return s, l, m, sp
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def DialogLoader(filename1, filename2, filename5, filename6, batch_size, shuffle):
    dataset = UtteranceDataset(filename1, filename2, filename5, filename6)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader