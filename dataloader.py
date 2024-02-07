import torch
import pandas as pd
import transformers
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

print("Loading Dataset")
df_train = pd.read_csv("train_new.csv", sep=",").dropna()
# df_train_pos = df_train[df_train["is_duplicate"]==0].head(119464)
# df_train_neg = df_train[df_train["is_duplicate"]==1]
# df_train = pd.concat([df_train_pos, df_train_neg])
df_train = df_train.sample(frac=1)

print(df_train['is_duplicate'].value_counts())
df_val_test = pd.read_csv("test_new.csv", sep=",").dropna()
df_valid = df_val_test.sample(n = 1)
print(df_valid['is_duplicate'].value_counts())
df_test = df_val_test.drop(df_valid.index)
# prinbt
# print((df_train))
# df_valid = pd.read_csv("../../../../../../data/spand43/unt/val.tsv", sep="\t", header=None)
# df_test = pd.read_csv("../../../../../../data/spand43/unt/test_PU_balanced.tsv", sep="\t")
# df_valid = df_valid[[0, 1, 3, 2]].rename(columns={0: "S.no", 1:"id", 3:"text", 2:"label" })
# df_train['label'].replace({'P': 1, 'U': 0},inplace=True)
# df_train = df_train[["S.no", "id", "text", "label"]]
# df_valid['label'].replace({'1.ForRepo': 1, '2.NotForRepo': 0},inplace=True)
# # df_test = df_test[[0, 1, 3, 2]].rename(columns={0: "S.no", 1:"id", 3:"text", 2:"label" })
# df_test = df_test[["0", "1", "3", "2"]].rename(columns={"0": "S.no", "1":"id", "3":"text", "2":"label" })
# df_test['label'].replace({'1.ForRepo': 1, '2.NotForRepo': 0},inplace=True)


class BertDataset(Dataset):
    def __init__(self, df, tokenizer,max_length):
        super(BertDataset, self).__init__()
        self.train_csv=df
        self.tokenizer=tokenizer
        self.target=self.train_csv.iloc[:,5]
        self.max_length=max_length
        
    def __len__(self):
        return len(self.train_csv)
    
    def __getitem__(self, index):
        text1 = self.train_csv.iloc[index,3]
        text2 = self.train_csv.iloc[index,4]  
        inputs = self.tokenizer.encode_plus(
            text1 ,
            text2,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.train_csv.iloc[index, 5]).type(torch.cuda.LongTensor)
            }

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
dataset= BertDataset(df_train, tokenizer, max_length=512)

train_loader=DataLoader(dataset=dataset,batch_size=16)
dataset= BertDataset(df_valid, tokenizer, max_length=512)

valid_loader=DataLoader(dataset=dataset,batch_size=16)

dataset= BertDataset(df_test, tokenizer, max_length=512)

test_loader=DataLoader(dataset=dataset,batch_size=16)