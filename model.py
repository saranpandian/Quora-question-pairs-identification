
import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import *


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")#.to("cuda")
        self.out1 = nn.Linear(768, 300)
        self.out = nn.Linear(300, 2)

        self.softmax = nn.Softmax()
        
    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        
        out= self.out1(o2)
        
        out= self.out(out)
        return out
    
