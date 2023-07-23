from transformers import set_seed, BertTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import argparse, torch, datasets, ast, operator, gc, os
print(f"### GPU available is {torch.cuda.is_available()} ###")
set_seed(2022)

    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='normal fewshot finetuning')
    parser.add_argument('--max_len', default=512, type=int, help="max sequence length")
    parser.add_argument('--lr', default=1e-6, type=float, help="learning rate")
    parser.add_argument('--epochs', default=20, type=int, help="number of epoch")
    parser.add_argument('--batch_size', default=2, type=int, help="batch size")
    parser.add_argument('--eval_batch_size', default=4, type=int, help="batch size")
    parser.add_argument('--model', default='./models/my_bert/', type=str, help="pretrained model dir")
    parser.add_argument('--output_dir', default='./ft_models', type=str, help="save tuned model dir")
    parser.add_argument('--result_output_dir', default='./ft_fewshot_results_equal/', type=str, help="exp result dir")
    parser.add_argument('--dataset_dir', default='./datasets/', type=str, help="dataset dir")
    parser.add_argument('--tokenizer', default='./models/my_bert/', type=str, help="tokenizer")
    parser.add_argument('--task', default='ecommerce', type=str, help="task name")
    parser.add_argument('--num_labels', default=2, type=int, help="task name")


    args = parser.parse_args()
    print(args)

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'loading tokenizer {args.tokenizer}')
    tokenizer = BertTokenizer.from_pretrained(f'{args.tokenizer}')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_len)


    accs = []
    

    seed = 'all'
    few_shot_data_path = f'{args.dataset_dir}{args.task}_output_{seed}.csv'
    print(f'#### read fewshot data from {few_shot_data_path} ####')
    df_train = pd.read_csv(f'{few_shot_data_path}',names=['labels','text'],header=0)
    df_dev = pd.read_csv(f'{args.dataset_dir}{args.task}_output_dev.csv',names=['labels','text'],header=0)

    dataset_train, dataset_eval = Dataset.from_pandas(df_train), Dataset.from_pandas(df_dev)
    zh_dataset = datasets.DatasetDict({"train":dataset_train,"eval":dataset_eval})

    tokenized_datasets = zh_dataset.map(tokenize_function, batched=True)

    for item in ['text','__index_level_0__']:
        tokenized_datasets = tokenized_datasets.remove_columns(item)

    tokenized_datasets.set_format("torch")


    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=False, batch_size=args.batch_size)
    eval_dataloader = DataLoader(tokenized_datasets['eval'], batch_size=args.eval_batch_size)


    #Activate the training mode of our model, and initialize our optimizer (Adam with weighted decay - reduces chance of overfitting).
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Now we can move onto the training loop and we eval in each loop.
    epochs = args.epochs
    current_dev_loss = np.inf
        
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        
        # epoch total loss
        epoch_loss = 0.0
        model.train()
        loop = tqdm(train_dataloader, leave=True)

        ## training
        for batch in train_dataloader:
            # activate training mode

            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
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

            del outputs, batch
            gc.collect()
            torch.cuda.empty_cache()

        

        saved_model_path = f'{args.output_dir}/{args.task}/{seed}'

        if not os.path.exists(f'{args.output_dir}/{args.task}'):
            os.mkdir(f'{args.output_dir}/{args.task}')

        print(f' ### saving model to {saved_model_path}###')
        model.save_pretrained(f'{saved_model_path}')
           

        gc.collect()
        torch.cuda.empty_cache()
                
        

    ## for testing 
    df_test = pd.read_csv(f'{args.dataset_dir}{args.task}_output_test.csv',names=['labels','text'],header=0)

    dataset_test = Dataset.from_pandas(df_test)
    test_dataset = datasets.DatasetDict({"test":dataset_test})

    tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)

    for item in ['text','__index_level_0__']:
        tokenized_test_datasets = tokenized_test_datasets.remove_columns(item)

    tokenized_test_datasets.set_format("torch")


    test_dataloader = DataLoader(tokenized_test_datasets['test'], shuffle=False, batch_size=args.eval_batch_size)
    load_path = f'{args.output_dir}/{args.task}/{seed}'
    print(f'### load pretrained model {load_path}')
    model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=args.num_labels).to(device)
    all_predictions = []
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions)

    labels = df_test['labels'].tolist()
    preds = [result.item() for result in all_predictions]
    acc = accuracy_score(labels, preds)
    print(acc)
    accs.append(acc)

    if not os.path.exists(args.result_output_dir):
        os.mkdir(args.result_output_dir)

    with open(f'{args.result_output_dir}{args.task}.txt','a') as out_file:
        out_file.write(f"acc_{seed}: {acc}\n")

print(f'avg test acc is {np.mean(accs)}')



