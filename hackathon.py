import os
import csv

import numpy as np
from scipy.special import softmax
import torch
from timm.utils import accuracy, AverageMeter
import tqdm

from data.dataset import HKDataset


def main():
    save_dir = '/Users/yhzhai/Downloads/save'
    dataset = HKDataset('dataset/metadata_test.csv',
                        'dataset',
                        'dataset/algo_map.json',
                        'dataset/label_map.json',
                        transform=None,
                        ignore_image=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    label_csv = open('task1.csv', 'w', newline='')
    label_csv_writer = csv.writer(label_csv)
    label_csv_writer.writerow(['challenge_id', 'ref_string', 'object_class',
                               'score'])
    algo_csv = open('task2.csv', 'w', newline='')
    algo_csv_writer = csv.writer(algo_csv)
    algo_csv_writer.writerow(['challenge_id', 'ref_string', 'algorithm_class',
                              'score'])
    mani_csv = open('task3.csv', 'w', newline='')
    mani_csv_writer = csv.writer(mani_csv)
    mani_csv_writer.writerow(['challenge_id', 'ref_string', 'manipulated',
                              'score'])
    label_acc = AverageMeter()
    algo_acc = AverageMeter()
    mani_acc = AverageMeter()
    for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataset)):
        challenge_id = data['id'][0]
        ref_string = data['ref_string'][0]
        label_gt = data['label']
        label_gt_name = dataset.label_map[str(label_gt.item())]
        algo_gt = data['algo']
        label_pred = np.load(os.path.join(save_dir,
                                          f'{challenge_id}_label.npy'))
        algo_pred = np.load(os.path.join(save_dir, f'{challenge_id}_algo.npy'))

        if label_gt_name != 'unmodified':
            # task 1
            obj_label_pred = torch.tensor(label_pred)[:-1].unsqueeze(0)
            obj_label_acc1 = accuracy(obj_label_pred, label_gt, topk=(1,))
            label_acc.update(obj_label_acc1[0].item())

            obj_label_pred = torch.softmax(obj_label_pred, dim=1)
            obj_pred_index = int(torch.argmax(obj_label_pred))
            obj_pred_name = dataset.label_map[str(obj_pred_index)]
            obj_pred_score = obj_label_pred[0, obj_pred_index].item()
            label_csv_writer.writerow([challenge_id, ref_string, obj_pred_name,
                                       obj_pred_score])
            
            # task 2
            algo_pred = torch.tensor(algo_pred)[:-1].unsqueeze(0)
            algo_acc1 = accuracy(algo_pred, algo_gt, topk=(1,))
            algo_acc.update(algo_acc1[0].item())

            algo_pred = torch.softmax(algo_pred, dim=1)
            algo_pred_index = int(torch.argmax(algo_pred))
            algo_pred_name = dataset.algo_map[str(algo_pred_index)]
            algo_pred_score = algo_pred[0, algo_pred_index].item()
            algo_csv_writer.writerow([challenge_id, ref_string, algo_pred_name,
                                      algo_pred_score])

        # task 3
        mani_label_pred = torch.tensor(label_pred)
        mani_label_pred = torch.softmax(mani_label_pred, dim=0)
        if ((mani_label_pred[-1] >= 0.5).all().item() and label_gt_name == 'unmodified') or \
                ((mani_label_pred[-1] <= 0.5).all().item() and label_gt_name != 'unmodified'):
            mani_acc.update(1)
        else:
            mani_acc.update(0)
        
        if (mani_label_pred[-1] >= 0.5).all().item():
            manipulated = 0
        else:
            manipulated = 1
        mani_csv_writer.writerow([challenge_id, ref_string, manipulated,
                                  1 - mani_label_pred[-1].item()])
    
    label_csv.close()
    algo_csv.close()
    mani_csv.close()

    print(label_acc.avg, algo_acc.avg, mani_acc.avg)


if __name__ == '__main__':
    main()
