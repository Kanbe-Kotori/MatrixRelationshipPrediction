import torch
import _2_Dataset as D
import _3_Model as M

from sklearn.metrics import average_precision_score


def calculate_auprc(predict, actual):
    predict = predict.detach().numpy()
    actual = actual.detach().numpy()
    auprc = average_precision_score(actual, predict, average='macro')
    return auprc


dataTrain = D.generateDataset(list(range(300)))
dataTest = D.generateDataset(list(range(300, 350)))

loaderTrain = dataTrain.getLoader(100)
loaderTest = dataTest.getLoader(100)

GPU: bool = torch.cuda.is_available()
model = M.Model()
# costFunc = torch.nn.BCELoss()
costFunc = M.ImbalancedLoss(alpha=19)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
if GPU:
    model.cuda()
    costFunc.cuda()

for epoch in range(1, 10001):
    tp, tn, fp, fn = 0, 0, 0, 0
    # auprcCount = 0
    lossCount = 0
    for (feature, label) in loaderTrain:
        model.train()
        if GPU:
            (feature, label) = (feature.cuda(), label.cuda())
        optimizer.zero_grad()
        modelInputs = feature.reshape(feature.shape[0], 7, 100)
        modelOutputs = model(modelInputs)
        loss = costFunc(modelOutputs, label)
        loss.backward()
        optimizer.step()

        output = modelOutputs.cpu()
        label = label.cpu()

        threshold = 0.5
        binaryOutputs = torch.where(output >= threshold, torch.tensor(1.), torch.tensor(0.))
        tp += ((binaryOutputs == 1) & (label == 1)).sum().item()  # true positives
        tn += ((binaryOutputs == 0) & (label == 0)).sum().item()  # true negatives
        fp += ((binaryOutputs == 1) & (label == 0)).sum().item()  # false positives
        fn += ((binaryOutputs == 0) & (label == 1)).sum().item()  # false negatives
        lossCount += loss

    if epoch % 1 == 0:
        accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
        recall = 100 * tp / (tp + fn)
        # auprcCount /= len(dataTrain) / 100
        auprcCount = 0
        print('epoch {} acc {:.3f}% recall {:.3f}% auprc {:.3f}%'.format(epoch, accuracy, recall, auprcCount))
        print('epoch {} loss {}'.format(epoch, lossCount / len(dataTrain)))
    if epoch % 10 == 0:
        model.eval()
        testtp, testtn, testfp, testfn = 0, 0, 0, 0
        for (feature, label) in loaderTest:
            if GPU:
                (feature, label) = (feature.cuda(), label.cuda())
            modelInputs = feature.reshape(feature.shape[0], 7, 100)
            modelOutputs = model(modelInputs)

            output = modelOutputs.cpu()
            label = label.cpu()

            threshold = 0.5
            binaryOutputs = torch.where(modelOutputs.cpu() >= threshold, torch.tensor(1.), torch.tensor(0.))
            testtp += ((binaryOutputs == 1) & (label == 1)).sum().item()  # true positives
            testtn += ((binaryOutputs == 0) & (label == 0)).sum().item()  # true negatives
            testfp += ((binaryOutputs == 1) & (label == 0)).sum().item()  # false positives
            testfn += ((binaryOutputs == 0) & (label == 1)).sum().item()  # false negatives
        accuracy = 100 * (testtp + testtn) / (testtp + testtn + testfp + testfn)
        recall = 100 * testtp / (testtp + testfn)
        print('test acc {:.3f}% recall {:.3f}%'.format(accuracy, recall))
    if epoch % 1000 == 0:
        torch.save(model.state_dict(), './model/' + str(epoch) + '.pkl')
