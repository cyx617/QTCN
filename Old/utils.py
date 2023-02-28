import torch
import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis
from sklearn.preprocessing import MinMaxScaler

import copy
import warnings
warnings.filterwarnings("ignore")


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# customized loss
def my_loss(output, target):
    loss = np.mean(((output - target)/(target))**2)
    return loss


#visualization of model prediction results
def plot_pred(factor,testX,testY,model,scaler,batch_size):

    labels = []
    preds = []
    #batch_size = 10
    model.eval()

    test_set = torch.utils.data.TensorDataset(torch.tensor(testX),torch.tensor(testY))
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size)
    for x, y in test_loader:

        output = model(x).detach().numpy()
        y = scaler.inverse_transform(y.reshape(output.shape[0],1)).reshape(-1).tolist()
        output = scaler.inverse_transform(output).reshape(-1).tolist()
        preds.extend(output)
        labels.extend(y)
    labels.reverse()
    preds.reverse()
    df = pd.DataFrame({
                   "preds" : preds,
                   "labels" : labels })
    #df.to_csv('./plot_data/' + str(factor)+'.csv')
    df.plot(y=["preds", "labels"],title=str(factor))
    plt.show()
