import os
from zipfile import ZipFile
import requests
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Download WDC datasets')
    parser.add_argument('-c', '--category', default='computers', type=str,
                    help='archive_name')                    
    parser.add_argument('-s', '--save_path', default='computers', type=str,
                    help='destination folder')                    
    return parser.parse_args()


def get_train(args):
    train_archive = f'{args.category}_train.zip'
    url_train = f'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/trainsets/{train_archive}'
    r = requests.get(url_train)
    open(train_archive, 'wb').write(r.content)

    print('Extracting...\n')
    with ZipFile(train_archive, 'r') as zip:
        zip.printdir()
        zip.extractall(args.save_path)
    os.remove(train_archive)


def get_test(args):
    test_archive = f'{args.category}_gs.json.gz'
    url_train = f'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/goldstandards/{test_archive}'
    r = requests.get(url_train)
    open(os.path.join(args.save_path, test_archive), 'wb').write(r.content)


if __name__ == '__main__':
    args = parse_args()
    print('Downloading the data...\n')
    get_train(args)
    get_test(args)
    print()
    print('Done!')




