import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import torch
from tqdm import tqdm
from torch import nn
from collections import OrderedDict
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import accuracy_score
import numpy as np

def prepare_data(df):
    CLASSES = list(df)[5:]
    if 'Unnamed: 0' in df.columns: df.drop('Unnamed: 0', axis=1, inplace=True)
    df['tags'] = df['tags'].replace(float('nan'), '{NONE}')
    df['text'] = df['text'].replace(float('nan'), '')
    # df['assessment'] = df['assessment'].replace(float('nan'), )
    TAGS = set([tag for tags in df['tags'].values for tag in tags[1:-1].split(',')])
    TAGS.remove('NONE')
    tags2id = OrderedDict()
    # tags2id['NONE'] = 0
    tags2id.update({v: k for k, v in enumerate(TAGS)})

    for tag in tags2id:
        df.insert(loc=4+tags2id[tag], column=tag, value=0)

    for row_idx, row in df.iterrows():
        row_tags = row['tags'][1:-1].split(',')
        for tag in row_tags:
            # if tag != 'NONE'
            df.loc[row_idx, tag]=1
            
    return df

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', string)
def remove_unwanted(document):

    # remove user mentions
    document = re.sub("@[A-Za-zА-я0-9_]+"," ", document)
    # remove URLS
    document = re.sub(r'http\S+', ' ', document)
    # remove hashtags
    document = re.sub("#[A-Za-zА-я0-9_]+"," ", document)
    # remove emoji's
    # document = remove_emoji(document)
    # remove punctuation
    document = re.sub("[^0-9A-Za-zА-я ]", " " , document)
    # remove double spaces
    document = document.replace('\s\s+'," ")
    
    return document.strip()

def remove_words(tokens):
    stopwords = nltk.corpus.stopwords.words('russian') # also supports german, spanish, portuguese, and others!
    stopwords = [remove_unwanted(word) for word in stopwords] # remove puntcuation from stopwords
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    return cleaned_tokens

lemma = WordNetLemmatizer()

def lemmatize(tokens):
    lemmatized_tokens = [lemma.lemmatize(token, pos = 'v') for token in tokens]
    return lemmatized_tokens

def pipeline(document, rule = 'lemmatize'):
    # first lets normalize the document
    #document = document.lower()
    document = document.replace('\n', ' ')
    # now lets remove unwanted characters
    document = remove_unwanted(document)
    # create tokens
    tokens = document.split()
    # remove unwanted words
    # tokens = remove_words(tokens)
    # tokens = lemmatize(tokens)
    
    return " ".join(tokens)

def get_classes(preds):
    class_preds = []
    for sample in preds:
        sample_class_preds = []
        for class_idx, _class in enumerate(sample):
            
            if _class == 1:
                sample_class_preds.append(class_idx)
        class_preds.append(' '.join([str(x) for x in sample_class_preds]))
    return class_preds

def train(model, train_dataloader, optimizer, loss_fn, EPOCHS, threshold=0.5, device='cuda:0', val_dataloader=None, val_data=None):
    losses = []
    # eval_losses = []
    for epoch in range(EPOCHS):
        model.train()
        for item in tqdm(train_dataloader):
            X_batch, y_batch = item['input'].to(device), item['label'].to(device)
            X_assist = None
            if 'tags' in item:
                
                X_assist = item['tags'].to(device)
                X_assist = X_assist.to(torch.float)
            # X_batch = X_batch.to(torch.float)
            y_batch = y_batch.to(torch.float)
            optimizer.zero_grad()
            output = model(X_batch, X_assist)
            loss = loss_fn(output.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        if val_data:
            preds = torch.tensor(eval(model, val_dataloader, threshold=threshold))
            for threshold in np.arange(0.5, 0.6, 0.1):
                print(f'threshold: {threshold}')    
                preds_thresh = (nn.functional.sigmoid(preds) > threshold).int()
                eval_acc = accuracy_score(val_data, preds_thresh.tolist())
                print(f'accuracy: {eval_acc}')
    return losses

def eval(model, test_dataloader, threshold=0.5, device='cuda:0'):
    model.eval()
    preds = []
    # y_true = [x[1] for x in list(test_data)]
    with torch.no_grad():
        for item in tqdm(test_dataloader):
            X_assist = None
            if 'tags' in item:
                X_assist = item['tags'].to(device)
                X_assist =X_assist.to(torch.float)
            X_batch = item['input'].to(device)
            # X_batch = X_batch.to(torch.float)
            output = model(X_batch, X_assist)
            preds.extend(output.float().detach().cpu().tolist())
    return preds