import random
from torch.utils.data import Dataset, DataLoader
import cv2
import re
import torch
import numpy as np
import einops


class DataGenerator(Dataset):
    def __init__(self, data, word2int):
        self.files = data
        self.word2int = word2int

    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        img_path, cap = self.files[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        return img, cap

    def collate_fn_customised(self, data):
        
        img_list = []
        cap_list = []
        for i, c in data:
            img_list.append(i)
            cap_list.append(c)
        
        ##OPERATING ON CAPTION
        modified_cap = []
        for cap in cap_list:
            #Removing Functuatoons and Numbers and converting it into lower case and breaking it into tokens.
            cap = re.sub('[^a-zA-Z ]+', '', cap.lower()).split()
            #Added BOS and EOS to the start and the end of the sent. respectively.
            cap = ["<BOS>"] + cap + ["<EOS>"] 
            modified_cap.append(cap)
        max_len = max([len(cap) for cap in modified_cap])
        cap_list = []
        for cap in modified_cap:
            while(len(cap)<max_len):
                cap = cap + ["<PAD>"]
            cap_list.append(cap)
        modified_cap = []
        for cap in cap_list:
            numericalizaed_sentences = []
            for word in cap:
                try:
                    numericalizaed_sentences.append(self.word2int[word])
                except:
                    numericalizaed_sentences.append(self.word2int["<UNK>"])                    
            modified_cap.append(numericalizaed_sentences)

        cap_tensor = torch.tensor(np.array(modified_cap))
        img_tensor = torch.tensor(np.array(img_list), dtype=torch.float32)
        img_tensor = einops.rearrange(img_tensor, "B H W C -> B C H W")
        cap_tensor = einops.rearrange(cap_tensor, "B T -> T B")
        return img_tensor, cap_tensor

def load_data(data, word2int, batch_size=1, num_workers=10, shuffle=True):

	dataset = DataGenerator(data, word2int)
	data_loader = DataLoader(dataset, collate_fn= dataset.collate_fn_customised, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return data_loader