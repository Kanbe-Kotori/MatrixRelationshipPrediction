import csv
import os
import numpy
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path

rootPath: str = os.path.join(Path(__file__).parent, '5mr')
DoNormalize = True


def relation(listIndex: list[int]):
    for index in listIndex:
        print('Table {}/{} :'.format(index, len(listIndex)))
        folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(index)

        fileNameData = 'simulated_noNoise_{}.txt'.format(index)
        fullPathData = os.path.join(rootPath, folder, fileNameData)
        dataLines = open(fullPathData).read().split('\n')[:100]
        data = numpy.array([line.split('\t') for line in dataLines]).astype(float).transpose()

        if DoNormalize:
            mean = numpy.mean(data, axis=1, keepdims=True)
            std = numpy.std(data, axis=1, keepdims=True)
            data = (data - mean) / std

        fileNameLabel = 'bipartite_GRN.csv'
        fullPathLabel = os.path.join(rootPath, folder, fileNameLabel)
        labelReader = csv.reader(open(fullPathLabel))
        labels = numpy.zeros([100, 100])
        for line in labelReader:
            if int(line[0]) < 5:
                continue
            tr = int(line[0]) - 5
            p = int(line[1]) - 105
            labels[p][tr] = 1

        dataTF = data[4:104, :]
        dataG = data[104:204, :]

        matrix = numpy.vstack((dataG, dataTF))
        cov = numpy.cov(matrix)
        covInv = numpy.linalg.inv(cov)
        pearson = numpy.corrcoef(matrix)
        spearman, _ = spearmanr(matrix, axis=1)
        mi = numpy.zeros([200, 200])

        for i in range(200):
            for j in range(i, 200):
                mi[i][j] = mi[j][i] = mutual_info_regression(matrix[i].reshape(-1, 1), matrix[j])

        fileNameData = 'feature.npz'
        savePath = os.path.join(rootPath, folder, fileNameData)
        numpy.savez(
            savePath,
            cov=cov,
            covInv=covInv,
            pearson=pearson,
            spearman=spearman,
            mi=mi,
            label=labels
        )


def hist2d(listIndex: list[int]):
    for index in listIndex:
        print('Table {}/{} :'.format(index, len(listIndex)))
        folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(index)

        fileNameData = 'simulated_noNoise_{}.txt'.format(index)
        fullPathData = os.path.join(rootPath, folder, fileNameData)
        dataLines = open(fullPathData).read().split('\n')[:100]
        data = numpy.array([line.split('\t') for line in dataLines]).astype(float).transpose()

        if DoNormalize:
            # mean = numpy.mean(data, axis=0)
            # std = numpy.std(data, axis=0)
            # data = (data - mean) / std
            data = numpy.log(data + numpy.e ** -3)

        fileNameLabel = 'bipartite_GRN.csv'
        fullPathLabel = os.path.join(rootPath, folder, fileNameLabel)
        labelReader = csv.reader(open(fullPathLabel))
        labels = numpy.zeros([100, 100])
        for line in labelReader:
            if int(line[0]) < 5:
                continue
            tr = int(line[0]) - 5
            p = int(line[1]) - 105
            labels[p][tr] = 1

        dataTF = data[4:104, :]
        dataG = data[104:204, :]

        label0 = []
        label1 = []
        for tf in range(100):
            for g in range(100):
                hist, _, _ = numpy.histogram2d(dataTF[tf], dataG[g], bins=16, density=True)
                hist = hist.reshape((1, 16, 16))
                label = int(labels[g][tf])
                if label == 0:
                    label0.append(hist)
                else:
                    label1.append(hist)

        label0 = numpy.concatenate(label0)
        label1 = numpy.concatenate(label1)
        print(label0.shape)
        print(label1.shape)
        savePath0 = os.path.join(rootPath, folder, '0.npy')
        savePath1 = os.path.join(rootPath, folder, '1.npy')
        numpy.save(savePath0, label0)
        numpy.save(savePath1, label1)


hist2d(list(range(350)))
