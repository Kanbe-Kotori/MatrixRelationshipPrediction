import csv
import os
import numpy
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path

rootPath: str = os.path.join(Path(__file__).parent, '5mr')
DoNormalize = True


def execute(listIndex: list[int]):
    for index in listIndex:
        print('Table {}/{} :'.format(index, len(listIndex)))
        folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(index)

        fileNameData = 'simulated_noNoise_{}.txt'.format(index)
        fullPathData = os.path.join(rootPath, folder, fileNameData)
        dataLines = open(fullPathData).read().split('\n')[:100]
        data = numpy.array([line.split('\t') for line in dataLines]).astype(float).transpose()
        
        if DoNormalize:
            mean = numpy.mean(data, axis=0)
            std = numpy.std(data, axis=0)
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

        for GType in range(100):
            listCov = numpy.zeros(5151)
            listCovInv = numpy.zeros(5151)
            listPearson = numpy.zeros(5151)
            listSpearman = numpy.zeros(5151)
            listMI = numpy.zeros(5151)
            matrix = numpy.vstack((dataG[GType], dataTF))
            cov = numpy.cov(matrix)
            covInv = numpy.linalg.inv(cov)
            pearson = numpy.corrcoef(matrix)
            spearman, _ = spearmanr(matrix, axis=1)

            current = 0
            for i in range(101):
                for j in range(i, 101):
                    listCov[current] = cov[i][j]
                    listCovInv[current] = covInv[i][j]
                    listPearson[current] = pearson[i][j]
                    listSpearman[current] = spearman[i][j]
                    listMI[current] = mutual_info_regression(matrix[i].reshape(-1, 1), matrix[j])
                    current += 1

            fileNameData = 'feature{}.npz'.format(GType)
            savePath = os.path.join(rootPath, folder, fileNameData)
            numpy.savez(
                savePath,
                cov=listCov,
                covInv=listCovInv,
                pearson=listPearson,
                spearman=listSpearman,
                mi=listMI,
                label=labels[GType]
            )
            if GType % 20 == 0:
                print(' GType {}/{}'.format(GType, 100))


execute(list(range(350)))
