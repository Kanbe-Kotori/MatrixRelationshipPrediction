import os
import csv
import numpy
import torch
import torch.utils.data as data
from pathlib import Path


rootPath: str = os.path.join(Path(__file__).parent, '5mr')


class Dataset(data.Dataset):
    def __init__(self, listData: list[tuple], transform=None):
        self.listData: list[tuple] = listData

    def __len__(self):
        return len(self.listData)

    def __getitem__(self, index):
        feature, label = self.listData[index]
        return torch.Tensor(feature), torch.Tensor(label)

    def getLoader(self, batchSize) -> data.DataLoader:
        return data.DataLoader(
            dataset=self,
            batch_size=batchSize,
            shuffle=True
        )


def readFromSingle(index: int):
    folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(index)
    dataPath = os.path.join(rootPath, folder, 'feature_norm.npz')
    labelPath = os.path.join(rootPath, folder, 'bipartite_GRN.csv')

    arrayData = numpy.load(dataPath)
    listCov = arrayData['cov']
    listCovInv = arrayData['covInv']
    listPearson = arrayData['pearson']
    listSpearman = arrayData['spearman']
    listMI = arrayData['mi']

    tfCov = arrayData['tfCov']
    tfCovInv = arrayData['tfCovInv']
    tfPearson = arrayData['tfPearson']
    tfSpearman = arrayData['tfSpearman']
    tfMI = arrayData['tfMI']

    labelReader = csv.reader(open(labelPath))
    labels = numpy.zeros([100, 100])
    for line in labelReader:
        if int(line[0]) < 5:
            continue
        tr = int(line[0]) - 5
        p = int(line[1]) - 105
        labels[p][tr] = 1

    dataPairs = []
    for i in range(100):
        feature = numpy.vstack((
            numpy.concatenate((listCov[i], tfCov)),
            numpy.concatenate((listCovInv[i], tfCovInv)),
            numpy.concatenate((listPearson[i], tfPearson)),
            numpy.concatenate((listSpearman[i], tfSpearman)),
            numpy.concatenate((listMI[i], tfMI))
        ))
        dataPair = feature, labels[i]
        dataPairs.append(dataPair)

    return dataPairs


def generateDataset(listIndex: list[int]):
    fullData = []
    for index in listIndex:
        print('{}/{}'.format(index, len(listIndex)))
        fullData += readFromSingle(index)
    dataset = Dataset(fullData)
    return dataset
