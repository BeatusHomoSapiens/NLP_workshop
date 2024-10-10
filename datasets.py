import torch
from torch.utils.data import Dataset
import re
import numpy as np
import utils

TAGS = [
    'ASSORTMENT',
    'CATALOG_NAVIGATION',
    'DELIVERY',
    'PAYMENT',
    'PRICE',
    'PRODUCTS_QUALITY',
    'PROMOTIONS',
    'SUPPORT',
]
CLASSES = [f'trend_id_res{i}' for i in range(50)]

class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, model_config, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.is_test = is_test
        self.model_config = model_config
        self.LABEL2ID = {'entailment': 1, 'not_entailment': 0}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # targets = self.df['target'][index]
        label_text = self.df['label_text'].iloc[index]
        text = self.df['text'].iloc[index]
        tags = self.df['tags'].iloc[index]
        label_descr = self.df['label_descr'].iloc[index]
        inp = self.tokenizer.encode_plus(
            text, label_descr, add_special_tokens=True, 
            truncation=True, return_tensors='pt', 
            padding='max_length', max_length=130
        )
        
        for key in inp:
            inp[key] = inp[key].squeeze()

        item = {
            'input': inp,
            'label': self.LABEL2ID[self.df['target'].iloc[index]]
        }
        return item


class CommentsDataset(Dataset):
    def __init__(self, df, tokenizer=False, is_test=False):
        # self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.text = df['text'].values
        self.tags = df[TAGS].values
        # self.assessment = df['assessment'].values
        self.labels = df[CLASSES].values if not is_test else None
        
    def __len__(self):
        return self.text.size
    
    def __getitem__(self, index):
        #X = self.preprocess_data(X)
        inp = self.text[index] # ' '.join(self.tags[index][1:-1].split(',')) + 
        if self.tokenizer: inp = self.tokenizer.encode_plus(
            inp,
            add_special_tokens=True,
            max_length=80,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input': inp,
            'tags': torch.tensor(self.tags[index]),        
        }
        if not self.is_test:
            item['label'] = self.labels[index]
        return item
    
    def preprocess_data(self, X):
        X = re.sub('\s\s+', ' ', X)
        return X
    
class CommentsTFIDFDataset(Dataset):
    def __init__(self, df, tfidf_texts, is_test=False):
        self.tfidf_texts = tfidf_texts
        self.tags = df[TAGS].values
        self.is_test = is_test
        classes = list(set(df.columns).intersection(set(CLASSES)))
        if not is_test:
            self.y = df[classes].values
        

    def __len__(self):
        return len(self.tfidf_texts)
    
    def __getitem__(self, index):
        #  self.assessment[index], 
        item = {
            'input': torch.tensor(np.hstack([self.tfidf_texts[index], self.tags[index]]), dtype=torch.float)
            #'input': self.tfidf_texts[index].astype(np.float32)
        }
        if not self.is_test:
            item['label'] = self.y[index]
        return item
    
class CommentTagsDataset(Dataset):
    def __init__(self, df, tokenizer=False, is_test=False):
        # self.df = df
        self.is_test = is_test
        self.text = df['text'].values
        self.tags = df[TAGS].values
        self.assessment = df['assessment'].values
        self.labels = df[CLASSES].values if not is_test else None
        
    def __len__(self):
        return self.text.size
    
    def __getitem__(self, index):
        item = {
            'input': torch.tensor(self.tags[index]),
        }
        if not self.is_test:
            item['label'] = self.labels[index]
        return item