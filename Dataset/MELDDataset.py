import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np




'''
label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
'''
class MELDDataset(Dataset):

    def __init__(self, train = True):
        
        audiopath = 'AudioFeature-6373/MELD/AudioFeatures.pkl'
        print(audiopath)
        visualpath = 'VisualFeature-512/MELD/20250424_054006_model_visual_features_dict_tbb0.9_poseCT0.85.pkl'
        print(visualpath)
        
        _, self.videoSpeakers, self.videoLabels, _, _, _, _, self.trainVid,\
        self.testVid, _ = pickle.load(open('Data/MELD/Speakers.pkl', 'rb'))

        self.videoText = pickle.load(open('Data/MELD/TextFeatures.pkl', 'rb'))
        self.videoAudio = pickle.load(open(audiopath, 'rb'))
        self.videoVisual = pickle.load(open('Data/MELD/VisualFeatures.pkl', 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)


    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(np.array(self.videoText[vid])),\
            torch.FloatTensor(np.array(self.videoAudio[vid])),\
                torch.FloatTensor(np.array(self.videoVisual[vid])),\
                    torch.FloatTensor(np.array(self.videoSpeakers[vid])),\
                        torch.FloatTensor(np.array([1] * len(self.videoLabels[vid]))),\
                            torch.LongTensor(np.array(self.videoLabels[vid]))


    def __len__(self):
        return self.len


    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        output = []
        for i in dat:
            temp = dat[i].values
            if i <= 3:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], padding_value = 0)) 
            elif i <= 4:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], True, padding_value = 0))
            elif i <= 5:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], True, padding_value = -1))

        return output


