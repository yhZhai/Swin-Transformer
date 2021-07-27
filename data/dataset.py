import os
from typing import List
import csv
import json

import numpy as np
import torch
from PIL import Image
import torchvision


class HKDataset(torch.utils.data.Dataset):
    def __init__(self, datalist, data_path, algo_map, label_map,
                 transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform

        self.datalist = []
        with open(datalist, 'r') as f:
            dataset = csv.DictReader(f)
            for row in dataset:
                self.datalist.append(row)
        
        with open(algo_map, 'r') as f:
            self.algo_map = json.load(f)
        
        with open(label_map, 'r') as f:
            self.label_map = json.load(f)

    def __getitem__(self, index):
        info = self.datalist[index]
        algo = info['algo']
        algo = int(self.algo_map[algo])

        label = info['object_class']
        label = int(self.label_map[label])

        image_path = os.path.join(self.data_path, info['filename'])
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'algo': algo, 'label': label}

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    dataset = Dataset('dataset/metadata_train.csv',
                      'dataset',
                      'dataset/algo_map.json',
                      'dataset/label_map.json',
                      transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for i, data in enumerate(dataloader):
        print(i)

    # algo = set()
    # label = set()
    # with open('dataset/metadata_train.csv', 'r') as f:
    #     dataset = csv.DictReader(f)
    #     for row in dataset:
    #         algo.add(row['algo'])
    #         label.add(row['object_class'])
    # algo = sorted(algo)
    # algo_dict = {}
    # for i, item in enumerate(algo):
    #     algo_dict[f'{i}'] = item
    #     algo_dict[item] = f'{i}'
    # label = sorted(label)
    # label_dict = {}
    # for i, item in enumerate(label):
    #     label_dict[f'{i}'] = item
    #     label_dict[item] = f'{i}'
    # with open('dataset/algo_map.json', 'w') as f:
    #     json.dump(algo_dict, f)
    #     
    # with open('dataset/label_map.json', 'w') as f:
    #     json.dump(label_dict, f)
    #     
    # print('done')
    