from transformers import set_seed, BertTokenizer, BertModel
from tqdm import tqdm
import argparse, torch
import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
from matplotlib import colors
import torch.nn.functional as f
import numpy as np
import matplotlib
font = {'size'   : 22}

matplotlib.rc('font', **font)
set_seed(2022)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='search for best template according to dev set')
    parser.add_argument('--max_len', default=512, type=int, help="max sequence length")
    parser.add_argument('--batch_size', default=2, type=int, help="batch size")
    parser.add_argument('--model', default='../models/my_bert/', type=str, help="pretrained model")
    parser.add_argument('--tokenizer', default='../models/my_bert/', type=str, help="tokenizer")
    parser.add_argument('--task', default='weibo', type=str, help="task name")
    parser.add_argument('--datasets', default='../datasets_ppl_score/', type=str, help="dataset dir")
    parser.add_argument('--template', default='很好。', type=str, help="template")
    parser.add_argument('--input_data', default='../datasets/', type=str, help="input data dir")
    args = parser.parse_args()

    device = 'cuda:0'

    tokenizer = BertTokenizer.from_pretrained(f'{args.tokenizer}')
    pretrained_model = BertModel.from_pretrained(args.model).to(device)

    pd_all = pd.read_csv(f'{args.input_data}{args.task}_output.csv',names=['labels','text'],header=0)
    texts = pd_all.text.tolist()
    labels = pd_all.labels.tolist()

    text_embeddings = []

    with torch.no_grad():
        for (text,label) in tqdm(zip(texts,labels)):
            
            if label == 1:
                text = '不满意。'+text
            else:
                text = '很满意。'+text

            # if label == 1:
            #     text = '黄黑。'+text
            # else:
            #     text = '绿黑。'+text

            # text = '[MASK]满意。'+text

            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = pretrained_model(**inputs)

            last_hidden_states = outputs.last_hidden_state.squeeze(0).mean(0)
            text_embeddings.append(last_hidden_states)

    text_embeddings = torch.stack(text_embeddings)
    norm_text_vectors = f.normalize(text_embeddings,p=2,dim=1).cpu()

    ## visualize
    manifold = umap.UMAP(n_neighbors=15,min_dist=0.0,random_state = 2022).fit(norm_text_vectors)
    X_reduced_2 = manifold.transform(norm_text_vectors)

    pos_index = [idx for idx,label in enumerate(labels) if label==1]
    neg_index = [idx for idx,label in enumerate(labels) if label!=1]



    cmap = colors.ListedColormap(['salmon'])
    cmap1 = colors.ListedColormap(['steelblue'])

    fig, ax = plt.subplots(figsize=(20,20))
    ax.scatter(X_reduced_2[pos_index][:, 0], X_reduced_2[pos_index][:, 1], c=np.array(labels)[pos_index], s=20,cmap=cmap,label='Positive')
    ax.scatter(X_reduced_2[neg_index][:, 0], X_reduced_2[neg_index][:, 1], c=np.array(labels)[neg_index],marker='s', s=20,cmap=cmap1,label='Negative')
    ax.legend(prop={'size': 40},markerscale=6)
    


    plt.savefig(f'{args.task}_w_reverse_template.pdf',bbox_inches='tight')