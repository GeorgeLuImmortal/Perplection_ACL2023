seed  = 1
from transformers import set_seed, BertTokenizer, TFAutoModelForMaskedLM, AutoTokenizer, get_scheduler, BertForPreTraining, BertForMaskedLM, BertConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import argparse, torch, datasets, ast, operator, gc, os
import torch.nn as nn
set_seed(seed)

def compute_ppl(sentence,model):
#     if len(sentence)>128:
#         sentence = sentence[:128]
    tokenize_input = tokenizer.tokenize(sentence)
    candidates = []
    target_words = []
    for i in range(len(tokenize_input)):
        temp = tokenize_input.copy()
        # temp = tokenize_input.copy()[:i+1]
        target_words.append(temp[i])
        temp[i] = tokenizer.mask_token
        candidates.append(temp)

        
    # print(candidates)
    with torch.no_grad():
        model.eval()     
        sentence_loss = 0.

        for i, (candidate, target_word) in enumerate(zip(candidates,target_words)):
            
            target_word_id = tokenizer.convert_tokens_to_ids(target_word)
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(candidate)]).to(device)
            output = model(tensor_input)
            prediction_scores = output.logits
            softmax = nn.Softmax(dim=0)
            
            ps = softmax(prediction_scores[0, i]).log()
            word_loss = ps[target_word_id]
            word_loss_new = word_loss.detach().cpu().numpy()
            sentence_loss += word_loss_new.item()
        #     print(word_loss_new)
        

        # print(sentence_loss)
        # print(sentence_loss/len(candidates))
        ppl = np.exp(-sentence_loss/len(candidates))
        del output,ps,word_loss, prediction_scores,tensor_input,tokenize_input
        gc.collect()
        torch.cuda.empty_cache()
        return ppl


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='search for best template according to dev set')
    parser.add_argument('--max_len', default=512, type=int, help="max sequence length")
    parser.add_argument('--batch_size', default=2, type=int, help="batch size")
    parser.add_argument('--model', default='./models/my_bert/', type=str, help="pretrained model")
    parser.add_argument('--result_output_dir', default='./ppl_results/', type=str, help="output directory")
    parser.add_argument('--tokenizer', default='./models/my_bert/', type=str, help="tokenizer")
    parser.add_argument('--task', default='douban', type=str, help="task name")
    parser.add_argument('--datasets', default='./datasets_ppl_score/', type=str, help="dataset dir")
    parser.add_argument('--template', default='很好。', type=str, help="template")
    parser.add_argument('--input_data', default='./datasets/', type=str, help="input data dir")
    args = parser.parse_args()

    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'loading tokenizer {args.tokenizer}')
    tokenizer = BertTokenizer.from_pretrained(f'{args.tokenizer}')
    pretrained_model = BertForMaskedLM.from_pretrained(args.model).to(device)

    pd_all = pd.read_csv(f'{args.input_data}{args.task}_output.csv',names=['labels','text'],header=0)
    texts = pd_all.text.tolist()
    labels = pd_all.labels.tolist()

    temp_texts = []
    for (label,text) in zip(labels,texts):

     
        new_text = args.template+text
        # if label == 0:
        #     new_text = args.template+text
        # else:
        #     new_text = '很好。'+text

        temp_texts.append(new_text)

    ppl_scores = []
    print(temp_texts[0])
    for text in tqdm(temp_texts):
        score = compute_ppl(text,pretrained_model)
        ppl_scores.append(score)

    pd_all['ppl'] = ppl_scores
    pd_all.to_csv(f'{args.datasets}{args.task}_ppl_temp_{args.template}.csv') ## bidirectional