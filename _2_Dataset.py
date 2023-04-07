import os
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


def readFromSingle(folderIndex: int, featureIndex: int):
    folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(folderIndex)
    dataPath = os.path.join(rootPath, folder, 'feature{}.npz'.format(featureIndex))

    arrayData = numpy.load(dataPath)
    listCov = arrayData['cov']
    listCovInv = arrayData['covInv']
    listPearson = arrayData['pearson']
    listSpearman = arrayData['spearman']
    listMI = arrayData['mi']
    label = arrayData['label']

    dataPairs = []
    for i in range(100):
        feature = numpy.vstack((
            listCov,
            listCovInv,
            listPearson,
            listSpearman,
            listMI
        ))
        dataPair = feature, label
        dataPairs.append(dataPair)

    return dataPairs


def generateDataset(listIndex: list[int]):
    fullData = []
    for index in listIndex:
        print('{}/{}'.format(index, len(listIndex)))
        for feature in range(100):
            fullData += readFromSingle(index, feature)
    dataset = Dataset(fullData)
    return dataset
