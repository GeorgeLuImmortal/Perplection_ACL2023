from transformers import set_seed, BertTokenizer, TFAutoModelForMaskedLM, AutoTokenizer, get_scheduler, BertForPreTraining, BertForMaskedLM, BertConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import argparse, torch, datasets, ast, operator, gc, os
print(f"### GPU available is {torch.cuda.is_available()} ###")




class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx].detach().clone() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
    
def convert_to_template(text_list: list, template: str, num_label: int=1) -> list:
    ## use global var tokenizer
    global tokenizer
    
    ## return a list of template text
    template_result = []
    if num_label==1:
        for text in text_list:
            template = template.split('*sep+*')[0]
            tep = template.replace('*cls*','')
            tep = tep.replace('*sent_0*',text)
            tep = tep.replace('*+sent_0*',' '+text)           
            tep = tep.replace('*mask*',tokenizer.mask_token)
            template_result.append(tep)
    else:
        for text in text_list:
            template = template.split('*sep+*')[0]
            tep = template.replace('*cls*','')
            tep = tep.replace('*sent_0*',text)
            tep = tep.replace('*+sent_0*',' '+text)           
            tep = tep.replace('*mask*',tokenizer.mask_token)
            tep = tep.replace('mask*',tokenizer.mask_token)
            template_result.append(tep)

    return template_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='prompt fewshot finetuning')
    parser.add_argument('--max_len', default=512, type=int, help="max sequence length")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--epochs', default=20, type=int, help="number of epoch")
    parser.add_argument('--batch_size', default=2, type=int, help="batch size")
    parser.add_argument('--eval_batch_size', default=4, type=int, help="batch size")
    parser.add_argument('--model', default='hfl/chinese-roberta-wwm-ext', type=str, help="pretrained model dir")
    parser.add_argument('--output_dir', default='./tuned_models', type=str, help="save tuned model dir")
    parser.add_argument('--result_output_dir', default='./fewshot_results/', type=str, help="exp result dir")
    parser.add_argument('--dataset_dir', default='./datasets/', type=str, help="dataset dir")
    parser.add_argument('--tokenizer', default='hfl/chinese-roberta-wwm-ext', type=str, help="tokenizer")
    parser.add_argument('--task', default='weibo', type=str, help="task name")
    parser.add_argument('--mapping', default= "{0:'不',1:'很'}" , type=str, help="label mapping")
    parser.add_argument('--num_label_words', default=1 , type=int, help="label mapping")
    parser.add_argument('--label_convertion_flag', default='False' , type=str, help="whether label convertion to digit")


    args = parser.parse_args()
    print(args)
    args.mapping = ast.literal_eval(args.mapping)
    label_convertion_flag = True if args.label_convertion_flag == 'True' else False  ## check if original label need to convert to digit True for convertion


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'loading tokenizer {args.tokenizer}')
    tokenizer = BertTokenizer.from_pretrained(f'{args.tokenizer}')
    mapping = args.mapping

    ## convert str label to digits when label is str
    if label_convertion_flag:
        str2index = {}
        for idx,key in enumerate(list(mapping.keys())):
            if not key.isdigit():
                str2index[key] = idx

        mapping = {}
        for key, value in args.mapping.items():
            mapping[str2index[key]] = value

    print(mapping)



   
    template = '*cls**mask*满意。*sent_0**sep+*\n'
    print(f'##### {template} #####')


    for seed in range(5):  
        set_seed(seed)
        few_shot_data_path = f'{args.dataset_dir}{args.task}_output_{seed}.csv'
        print(f'#### read fewshot data from {few_shot_data_path} ####')
        df_train = pd.read_csv(f'{few_shot_data_path}',names=['labels','text'],header=0)
        df_dev = pd.read_csv(f'{args.dataset_dir}{args.task}_output_dev.csv',names=['labels','text'],header=0)

    
        train_text = convert_to_template(df_train['text'],template, args.num_label_words)
        df_train['text'] = train_text

        dev_text = convert_to_template(df_dev['text'],template, args.num_label_words)
        df_dev['text'] = dev_text

        if label_convertion_flag:
            df_train['labels'] = [str2index[label] for label in df_train['labels'].tolist()]
            df_dev['labels'] = [str2index[label] for label in df_dev['labels'].tolist()]
        
        print('### an example of template text ###')
        print(train_text[0])

        
        train_inputs = tokenizer(df_train['text'].tolist(), return_tensors='pt', max_length=args.max_len, truncation=True, padding='max_length')
        train_labels = train_inputs.input_ids.detach().clone()

        dev_inputs = tokenizer(df_dev['text'].tolist(), return_tensors='pt', max_length=args.max_len, truncation=True, padding='max_length')
        dev_labels = dev_inputs.input_ids.detach().clone()

        ## label is the id of template text with label tokens other token -100.
        for (label,label_input) in zip(df_train['labels'],train_labels):
            label_words = mapping[label]
              
            ids = torch.tensor(tokenizer.encode(label_words,add_special_tokens=False))
            label_input[label_input!=tokenizer.mask_token_id] = -100
            label_input[label_input==tokenizer.mask_token_id] = ids
        for (label,label_input) in zip(df_dev['labels'],dev_labels):
            label_words = mapping[label]
              
            ids = torch.tensor(tokenizer.encode(label_words,add_special_tokens=False))
            label_input[label_input!=tokenizer.mask_token_id] = -100
            label_input[label_input==tokenizer.mask_token_id] = ids
            

        train_inputs['labels'] = train_labels
        dev_inputs['labels'] = dev_labels


        ## Initialize our data using the `MyDataset` class.
        train_dataset = MyDataset(train_inputs)
        ## And initialize the dataloader, which we'll be using to load our data into the model during training.
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    
        dev_dataset = MyDataset(dev_inputs)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)


        # Now we can move onto setting up the training loop. First we setup GPU/CPU usage.
        print(f'loading model {args.model}')
        path = args.model
        model = BertForMaskedLM.from_pretrained(path).to(device)
    #     model = torch.nn.DataParallel(BertForMaskedLM.from_pretrained(path))

        #Activate the training mode of our model, and initialize our optimizer (Adam with weighted decay - reduces chance of overfitting).
        optimizer = AdamW(model.parameters(), lr=args.lr)

        # Now we can move onto the training loop and we eval in each loop.

        epochs = args.epochs
        current_dev_loss = np.inf 
        
        
        
        for epoch in range(epochs):
            # setup loop with TQDM and dataloader
            
            # epoch total loss
            epoch_loss = 0.0
            model.train()
            loop = tqdm(train_loader, leave=True)

            ## training
            for batch in loop:
                # activate training mode

                # initialize calculated gradients (from prev step)
                optimizer.zero_grad()
                # pull all tensor batches required for training
                input_ids = batch['input_ids'].to(device)

                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                # process
                outputs = model(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=labels)
                # extract loss
                loss = outputs.loss
                
                epoch_loss += loss
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optimizer.step()
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

                del outputs, token_type_ids, attention_mask, input_ids
                gc.collect()
                torch.cuda.empty_cache()

            saved_model_path = f'{args.output_dir}/{args.task}/{seed}'
            if not os.path.exists(f'{args.output_dir}/{args.task}'):
                    os.mkdir(f'{args.output_dir}/{args.task}')

            print(f' ### saving model to {saved_model_path}###')
            model.save_pretrained(f'{saved_model_path}')


        del model
        gc.collect()
        torch.cuda.empty_cache()
                
        


            
    df_test = pd.read_csv(f'{args.dataset_dir}{args.task}_output_test.csv',index_col=0,names=['labels','text'], header=0)

    if label_convertion_flag:
        df_test['labels'] = [str2index[label] for label in df_test['labels'].tolist()]

    test_labels = df_test['labels'].tolist()
    test_text = convert_to_template(df_test['text'],template,args.num_label_words)
    df_test['text'] = test_text

    test_inputs = tokenizer(df_test['text'].tolist(), return_tensors='pt', max_length=args.max_len, truncation=True, padding='max_length')
    test_dataset = MyDataset(test_inputs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    

    accs = []
    for seed in range(5):
        pred_joint_probs = []
        load_path = f'{args.output_dir}/{args.task}/{seed}'
        print(f'### load pretrained model {load_path}')
        model = BertForMaskedLM.from_pretrained(load_path).to(device)
        model.eval()
        for batch in tqdm(test_loader):

            input_ids = batch['input_ids']

            if args.num_label_words != 1:
                mask_token_ids = [np.argwhere(input_id == tokenizer.mask_token_id).squeeze() for input_id in input_ids]
            else:
                mask_token_ids = [np.argwhere(input_id == tokenizer.mask_token_id).item() for input_id in input_ids]

            input_ids = input_ids.to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # process
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)

            prediction_logits = outputs.logits.detach().cpu().numpy()

            if args.num_label_words!=1:
                ## softmax logits to probs
                sm = torch.nn.Softmax(dim=2)
                prediction_probs = sm(torch.tensor(prediction_logits))

                ## compute joint probablity
                for mask_token_id,pred in zip(mask_token_ids,prediction_probs):

                    pred_joint_prob = {} ## dict of label: score
                    for key,label_word in mapping.items():
                        ids = tokenizer.encode(label_word,add_special_tokens=False)

                        score_0 = pred[mask_token_id[0]][ids[0]]
                        score_1 = pred[mask_token_id[1]][ids[1]]
                        score = score_0*score_1
                        pred_joint_prob[key] = score

                    pred_joint_probs.append(pred_joint_prob)
            else:
                ## do not need to compute joint probablity
                for mask_token_id,pred in zip(mask_token_ids,prediction_logits):
                    pred_joint_prob = {}
                    for key,label_word in mapping.items():
                        ids = tokenizer.encode(label_word,add_special_tokens=False)
                        score = pred[mask_token_id][ids]
                        pred_joint_prob[key] = score
                        
                    pred_joint_probs.append(pred_joint_prob)


            del outputs, token_type_ids, attention_mask, input_ids
            gc.collect()
            torch.cuda.empty_cache()

        preds = [max(prob_dict, key=prob_dict.get) for prob_dict in pred_joint_probs] ## get label words with max joint probablity
        acc = accuracy_score(test_labels, preds)
        print(acc)
        accs.append(acc)

        if not os.path.exists(args.result_output_dir):
            os.mkdir(args.result_output_dir)

        with open(f'{args.result_output_dir}{args.task}.txt','a') as out_file:
            out_file.write(f"acc_{seed}: {acc}\n")

    print(f'avg test acc is {np.mean(accs)}')



