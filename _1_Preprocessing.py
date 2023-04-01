import os
import numpy
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path

rootPath: str = os.path.join(Path(__file__).parent, '5mr')
DoNormalize = True


def execute(listIndex: list[int]):
    for index in listIndex:
        print('{}/{}'.format(index, len(listIndex)))
        folder = 'De-noised_100G_9T_300cPerT_4_DS{}'.format(index)
        fileName = 'simulated_noNoise_{}.txt'.format(index)
        fullPath = os.path.join(rootPath, folder, fileName)
        dataLines = open(fullPath).read().split('\n')[:100]
        data = numpy.array([line.split('\t') for line in dataLines]).astype(float).transpose()
        
        if DoNormalize:
            mean = numpy.mean(data, axis=0)
            std = numpy.std(data, axis=0)
            data = (data - mean) / std
        
        dataTF = data[4:104, :]
        dataG = data[104:204, :]

        mutCov = numpy.zeros([100, 100])
        mutCovInv = numpy.zeros([100, 100])
        mutPearson = numpy.zeros([100, 100])
        mutSpearman = numpy.zeros([100, 100])
        mutMI = numpy.zeros([100, 100])
        for i in range(100):
            targetG = dataG[i]
            for j in range(100):
                cov = numpy.cov(targetG, dataTF[j])
                covInv = numpy.linalg.inv(cov)
                mutCov[i][j] = cov[0, 1]
                mutCovInv[i][j] = covInv[0, 1]
                mutPearson[i][j], _ = pearsonr(targetG, dataTF[j])
                mutSpearman[i][j], _ = spearmanr(targetG, dataTF[j])
                # Discrete calculation methods cannot be used
                # https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
                # bins = 8 -> Sturges' rule
                # c_xy = numpy.histogram2d(targetG, dataTF[j], 8)[0]
                # g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
                # mutMI[i][j] = 0.5 * g / c_xy.sum() / numpy.log(2)
                mutMI[i][j] = mutual_info_regression(targetG.reshape(-1, 1), dataTF[j])

        selfCov = numpy.zeros(4950)
        selfCovInv = numpy.zeros(4950)
        selfPearson = numpy.zeros(4950)
        selfSpearman = numpy.zeros(4950)
        selfMI = numpy.zeros(4950)
        current = 0
        for i in range(100):
            targetTF = dataTF[i]
            for j in range(i+1, 100):
                cov = numpy.cov(targetTF, dataTF[j])
                covInv = numpy.linalg.inv(cov)
                selfCov[current] = cov[0, 1]
                selfCovInv[current] = covInv[0, 1]
                selfPearson[current], _ = pearsonr(targetTF, dataTF[j])
                selfSpearman[current], _ = spearmanr(targetTF, dataTF[j])
                selfMI[current] = mutual_info_regression(targetTF.reshape(-1, 1), dataTF[j])
                current += 1

        fileName = 'feature_norm.npz' if DoNormalize else 'feature.npz'
        savePath = os.path.join(rootPath, folder, fileName)
        numpy.savez(savePath, cov=mutCov, covInv=mutCovInv, pearson=mutPearson, spearman=mutSpearman, mi=mutMI,
                    tfCov=selfCov, tfCovInv=selfCovInv, tfPearson=selfPearson, tfSpearman=selfSpearman, tfMI=selfMI)


execute(list(range(350)))
