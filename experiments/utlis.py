import os
from pathlib import Path
import requests
from zipfile import ZipFile

import numpy as np
import pandas as pd

import torch


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