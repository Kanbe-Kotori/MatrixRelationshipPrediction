import os
import numpy
import numpy as np
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
        return torch.Tensor(feature), label

    def getLoader(self, batchSize) -> data.DataLoader:
        return data.DataLoader(
            dataset=self,
            batch_size=batchSize,
            shuffle=True
        )


def readFromSingle(folderIndex: int):
    folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(folderIndex)
    dataPath = os.path.join(rootPath, folder, 'feature.npz')

    dataPairs = []
    with numpy.load(dataPath) as arrayData:
        listCov = arrayData['cov']
        listCovInv = arrayData['covInv']
        listPearson = arrayData['pearson']
        listSpearman = arrayData['spearman']
        listMI = arrayData['mi']
        label = arrayData['label']

        listTFiTFi = np.zeros([2, 100])
        for i in range(100):
            listTFiTFi[0][i] = listCov[i + 100][i + 100]
            listTFiTFi[1][i] = listMI[i + 100][i + 100]

        listTFiTFj = np.zeros([5, 5050])
        current = 0
        for i in range(100):
            for j in range(i, 100):
                listTFiTFj[0][current] = listCov[i + 100][j + 100]
                listTFiTFj[1][current] = listCovInv[i + 100][j + 100]
                listTFiTFj[2][current] = listPearson[i + 100][j + 100]
                listTFiTFj[3][current] = listSpearman[i + 100][j + 100]
                listTFiTFj[4][current] = listMI[i + 100][j + 100]
                current += 1

        for i in range(100):
            feature = numpy.vstack((
                listCov[i, 100:],
                listCovInv[i, 100:],
                listPearson[i, 100:],
                listSpearman[i, 100:],
                listMI[i, 100:],
            ))
            dataPair = np.concatenate((feature, listTFiTFj), axis=1), label[i]
            dataPairs.append(dataPair)

    return dataPairs


def histFromSingle(folderIndex: int):
    folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(folderIndex)
    path0 = os.path.join(rootPath, folder, '0.npy')
    path1 = os.path.join(rootPath, folder, '1.npy')
    data0 = np.load(path0)
    data1 = np.load(path1)
    dataPairs = []
    for i in range(data0.shape[0]):
        dataPairs.append((data0[i], 0))
    for i in range(data1.shape[0]):
        dataPairs.append((data1[i], 1))
    return dataPairs


def generateDataset(listIndex: list[int]):
    fullData = []
    for index in listIndex:
        print('{}/{}'.format(index, len(listIndex)))
        fullData += histFromSingle(index)
    dataset = Dataset(fullData)
    return dataset
