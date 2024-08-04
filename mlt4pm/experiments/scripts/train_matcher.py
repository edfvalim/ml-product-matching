import os
import sys
import datetime
import argparse
from pathlib import Path
import requests
from zipfile import ZipFile

import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import ProgressCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from google.colab import drive


class PolishDatasetLoader:
    MAIN_DIR_PATH = 'https://github.com/WilyLynx/mlt4pm/raw/master/data/PolishDataset'

    @staticmethod
    def load_train(type:object, size:object)->pd.DataFrame:
        """Loads the training dataset from repository

        Args:
            type (object): dataset type: all, chemia, napoje
            size (object): dataset size: small, medium, large

        Returns:
            pd.DataFrame: training dataset
        """
        path = f'{PolishDatasetLoader.MAIN_DIR_PATH}/{type}_train/pl_wdc_{type}_{size}.json.gz'
        df = pd.read_json(path, compression='gzip', lines=True)
        return df.reset_index()

    @staticmethod
    def load_test(type:object)->pd.DataFrame:
        """Loads the test dataset form repository

        Args:
            type (object): dataset type: all, chemia, napoje

        Returns:
            pd.DataFrame: test dataset
        """
        path = f'{PolishDatasetLoader.MAIN_DIR_PATH}/test/pl_wdc_{type}_test.json.gz'
        df = pd.read_json(path, compression='gzip', lines=True)
        return df.reset_index()


class EnglishDatasetLoader:
    MAIN_DIR_PATH = 'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2'

    @staticmethod
    def load_train(type:object, size:object)->pd.DataFrame:
        """Loads the training dataset from WDC website

        Args:
            type (object): dataset type: computers, cameras, watches, shoes, all
            size (object): dataset size: small, medium, large, xlarge

        Returns:
            pd.DataFrame: training dataset
        """
        p = Path(os.path.join('wdc_train', f'{type}_train'))
        p.mkdir(parents=True, exist_ok=True)
        dataset_path = f'{p}/{type}_train_{size}.json.gz'
        if not os.path.exists(dataset_path):
            zip_path = f'{p}.zip'
            url = f'{EnglishDatasetLoader.MAIN_DIR_PATH}/trainsets/{type}_train.zip'
            r = requests.get(url, allow_redirects=True)
            open(zip_path, 'wb').write(r.content)
            with ZipFile(zip_path, 'r') as zip:
                zip.extractall(path=p)
        
        df = pd.read_json(dataset_path, compression='gzip', lines=True)
        return df.reset_index()

    @staticmethod
    def load_test(type:object)->pd.DataFrame:
        """Loads the test dataset form repository

        Args:
            type (object): dataset type: computers, cameras, watches, shoes, all

        Returns:
            pd.DataFrame: test dataset
        """
        path = f'{EnglishDatasetLoader.MAIN_DIR_PATH}/goldstandards/{type}_gs.json.gz'
        df = pd.read_json(path, compression='gzip', lines=True)
        return df.reset_index()


class FeatureBuilder:
    def __init__(self, columns):
        self.columns = columns

    def get_X(self, dataset):
        X = '[CLS] ' + dataset[f'{self.columns[0]}_left']
        for i in range(1, len(self.columns)):
            X = X + ' [SEP] ' + dataset[f'{self.columns[i]}_left']
        for i in range(len(self.columns)):
            X = X + ' [SEP] ' + dataset[f'{self.columns[i]}_right']
        X + ' [SEP]'
        return X.to_list()

    def get_y(self, dataset):
        return dataset['label'].to_list()


class TorchPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.items = self.preprocessItems(encodings, labels)

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.labels)

    def preprocessItems(self, encodings, labels):
        items = []
        for idx in range(len(labels)):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            items.append(item)
        return items


def preprocess_train_val(data, feature_builder, tokenizer, random_seed, split_ratio=0.2,):
    X_train = feature_builder.get_X(data)
    y_train = feature_builder.get_y(data)
    if not random_seed:
        random_seed = np.random.RandomState()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_ratio, random_state=random_seed)
    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    val_encodings = tokenizer(X_val, truncation=True, padding=True)
    train_dataset = TorchPreprocessedDataset(train_encodings, y_train)
    val_dataset = TorchPreprocessedDataset(val_encodings, y_val)
    return train_dataset, val_dataset


def preprocess_test(data, feature_builder, tokenizer):
    X_test = feature_builder.get_X(data)
    y_test = feature_builder.get_y(data)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)
    return TorchPreprocessedDataset(test_encodings, y_test)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_model(model_name, dataset_type, dataset_size, lang, google, random_seed):
    print(f'BEGIN  EXPERIMENT')
    print(f'model: {model_name}')
    print(f'dataset: {dataset_type}')
    print(f'size: {dataset_size}')
    if google:
        drive.mount('/content/drive')

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    title_fb = FeatureBuilder(['title'])

    if lang == 'PL':
        dataset_loader = PolishDatasetLoader
    elif lang == 'ENG':
        dataset_loader = EnglishDatasetLoader

    train_df = dataset_loader.load_train(dataset_type, dataset_size)
    train_dataset, val_dataset = preprocess_train_val(train_df, title_fb, tokenizer, random_seed)

    test_df = dataset_loader.load_test(dataset_type)
    test_dataset = preprocess_test(test_df, title_fb, tokenizer)

    logdir_name = f'{model_name}_{dataset_type}_{dataset_size}'
    logdir = os.path.join("logs", logdir_name)
    train_batch_size = 16
    num_train_epochs = 10
    half_train = (len(train_dataset) * num_train_epochs) // (2*train_batch_size)
    if not random_seed:
        train_seed = np.random.randint(1_000_000)
    else:
        train_seed = random_seed
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=num_train_epochs,              # total number of training epochs
        per_device_train_batch_size=train_batch_size,   # batch size per device during training
        per_device_eval_batch_size=64,    # batch size for evaluation
        warmup_steps=half_train,                 # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                # strength of weight decay
        logging_dir=logdir,               # directory for storing logs
        logging_steps=10,                 # for training metrics
        disable_tqdm=False,               # show some progress
        fp16=True,                        # float 16 acceleration
        evaluation_strategy='epoch',      # evaluate after epoch
        load_best_model_at_end =True,     # load best model
        metric_for_best_model='eval_f1',  # use model with best F1 score
        save_total_limit=5,               # store last 5 checkpoints
        seed=train_seed                   # random seed
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    print('DEVICE USED: ', training_args.device)
    print('TRAINING')
    trainer.train()
    print('EVALUATION ON VALIDATION SET')
    eval = trainer.evaluate()
    print(eval)

    print('TESTING')
    pred = trainer.predict(test_dataset)
    metrics = pd.DataFrame(compute_metrics(pred), index=[0])
    metrics['model'] = model_name
    metrics['dataset_type'] = dataset_type
    metrics['dataset_size'] = dataset_size
    with pd.option_context('display.max_columns', None):
        print(metrics)

    print('SAVING MODEL')
    model_tmp_save = 'results/test'
    model.save_pretrained(model_tmp_save)
    if google:
        DRIVE = 'drive/MyDrive'
        p = Path(os.path.join(DRIVE, 'MLT4PM', lang, model_name, dataset_type, dataset_size))
        p.mkdir(parents=True, exist_ok=True)
        os.system(f'python -m transformers.convert_graph_to_onnx --model {model_tmp_save} --framework pt --tokenizer {model_name} {p}/model/model.onnx')
        os.system(f'mv {p}/model/model.onnx {p}/model.onnx')
        os.system(f'rm -R {p}/model/')
        os.system(f'rm -R {model_tmp_save}')
        metrics.to_csv(f'{p}/metrics.csv')

        log_path = Path(os.path.join(DRIVE, "MLT4PM", lang, "logs", logdir_name))
        log_path.mkdir(parents=True, exist_ok=True)
        os.system(f'cp -R {logdir} {log_path}')
    os.system('rm -R ./results')
    print('EXPERIMENT ENDED \n\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Fine tune transformer on PL WDC dataset')
    parser.add_argument('-m', '--model', default='bert-base-multilingual-uncased', type=str,
                    help='pretrained model available on huggingface model hub')
    parser.add_argument('-d', '--dataset', default='chemia', type=str,
                    help='dataset type: (chemia, napoje, all, cameras, computers, shoes, watches)  ')
    parser.add_argument('-s', '--size', default='small', type=str,
                    help='dataset size: (small, medium, large)')
    parser.add_argument('-g', '--google', default=False, type=bool,
                    help='True - save results to Google Drive')                    
    parser.add_argument('-l', '--lang', default='PL', type=str,
                    help='language of the datasets: PL, ENG')
    parser.add_argument('-rs', '--random_seed', default=None, type=int,
                    help='Random seed')             
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args.model, args.dataset, args.size, args.lang, args.google, args.random_seed)
