import os
import numpy
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path

rootPath: str = os.path.join(Path(__file__).parent, '5mr')


def execute(listIndex: list[int]):
    for index in listIndex:
        print('{}/{}'.format(index, len(listIndex)))
        folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(index)
        fileName = 'simulated_noNoise_{}.txt'.format(index)
        fullPath = os.path.join(rootPath, folder, fileName)
        dataLines = open(fullPath).read().split('\n')[:100]
        data = numpy.array([line.split('\t') for line in dataLines]).astype(float).transpose()
        dataTF = data[4:104, :]
        dataG = data[104:204, :]

        listCov = numpy.zeros([100, 100])
        listCovInv = numpy.zeros([100, 100])
        listPearson = numpy.zeros([100, 100])
        listSpearman = numpy.zeros([100, 100])
        listMI = numpy.zeros([100, 100])
        for i in range(100):
            targetG = dataG[i]
            for j in range(100):
                cov = numpy.cov(targetG, dataTF[j])
                covInv = numpy.linalg.inv(cov)
                listCov[i][j] = cov[0, 1]
                listCovInv[i][j] = covInv[0, 1]
                listPearson[i][j], _ = pearsonr(targetG, dataTF[j])
                listSpearman[i][j], _ = spearmanr(targetG, dataTF[j])
                # Discrete calculation methods cannot be used
                # https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
                # bins = 8 -> Sturges' rule
                # c_xy = numpy.histogram2d(targetG, dataTF[j], 8)[0]
                # g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
                # listMI[i][j] = 0.5 * g / c_xy.sum() / numpy.log(2)
                listMI[i][j] = mutual_info_regression(targetG.reshape(-1, 1), dataTF[j])

        savePath = os.path.join(rootPath, folder, 'feature.npz')
        numpy.savez(savePath, cov=listCov, covInv=listCovInv, pearson=listPearson, spearman=listSpearman, mi=listMI)


execute(list(range(350)))
