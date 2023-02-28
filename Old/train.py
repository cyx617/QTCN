import time
import os



import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

import copy

import warnings
warnings.filterwarnings("ignore")


from data import decomposition,normalization,data_preparation
from models import QTCN
from utils import my_loss,plot_pred





# Model training function
def train_loop(trainX,trainY, model, loss_func, optimizer,batch_size):

    total_loss = []
    model.train()


    n_train = 0
    train_loss = 0



    train_set = torch.utils.data.TensorDataset(torch.tensor(trainX),torch.tensor(trainY))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)




    for x, y in train_loader:

        optimizer.zero_grad()
        output = model(x)
        #print(y)
        #print(output)

        loss = loss_func(output.float(),y.float())
        #print(loss)
        loss_np = np.array(loss.item())


        loss.backward()
        optimizer.step()
        train_loss += loss_np
        n_train += 1




    loss =  train_loss / n_train


    return loss



# Model testing function
def test_loop(testX,testY, model, loss_func,my_loss_func,scaler,batch_size):


    total_loss = []
    model.eval()


    n_test = 0
    test_loss = 0
    test_my_loss = 0
    test_set = torch.utils.data.TensorDataset(torch.tensor(testX),torch.tensor(testY))
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=10)
    for x, y in test_loader:

        output = model(x).detach().numpy()

        y = y.reshape(output.shape[0],1)


        output_inversed = scaler.inverse_transform(output)
        y_inversed = scaler.inverse_transform(y)
        loss = loss_func(torch.tensor(output_inversed),torch.tensor(y_inversed))
        my_loss = my_loss_func(output_inversed,y_inversed)

        test_loss +=loss
        test_my_loss += my_loss
        n_test += 1

    loss =  test_loss / n_test

    acc = 1 - np.sqrt ((test_my_loss / n_test))


    return loss,acc,model

# 实验函数，包括数据生成、模型训练和测试
# factor_num是气象指标编号，取 0,1,2,3,4，分别对应大气压强，最低温度，最高温度，相对湿度和风速
# factor_num = 0,1,2,3,4, corresponding to 'Atmospheric Pressure','Minimum Temperature','Maximum Temperature','Relative Humidity','Wind Speed' respectively
def run_experiment(device,inputSize,window_size,factor_num,epochs,lr,batch_size_train,batch_size_test,verbose):

    writer = SummaryWriter()

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./plot_data"):
        os.makedirs("./plot_data")

    trainfile = 'data/Train.csv'
    testfile = 'data/Test.csv'
    factors = list(pd.read_csv(trainfile))[1:]
    print('Start training model for ' + str(factors[factor_num]))

    # SSA decomposition
    df_train_ssa,df_test_ssa = decomposition(trainfile,testfile,window_size,factors[factor_num])

    # minmax normalization
    df_train,df_test,scaler_dic = normalization(df_train_ssa,df_test_ssa)
    scaler = scaler_dic[factors[factor_num]]

    # generate traing and testing data
    trainXDt,trainX2Dt, trainYDt = data_preparation(df_train,inputSize,window_size)
    testXDt,testX2Dt, testYDt = data_preparation(df_test,inputSize,window_size)


    model = QTCN(device=device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_func_train = torch.nn.MSELoss()
    loss_func_test = my_loss



    epoch_list = []
    best_test_acc = 0

    for i in range(epochs):

        if i==0:
            start = time.time()

        train_loss = train_loop(trainXDt, trainYDt,model,loss_func_train,optimizer,batch_size_train)
        #print("{:.0f} train loss is : {:.10f}".format(i, train_loss))
        writer.add_scalar('Loss/train', train_loss, i)


        test_loss,test_acc,model_ = test_loop(testXDt, testYDt,model,loss_func_train,loss_func_test,scaler,batch_size_test)
        writer.add_scalar('Loss/test', test_loss, i)
        writer.add_scalar('Accuracy/test', test_acc, i)
        if verbose:
            print("{:.0f} test acc is : {:.10f}, test loss is : {:.10f}".format(i, test_acc,test_loss))


        #save the model with best test acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            #torch.save(model_.state_dict(),"./checkpoints/" + str(factors[factor_num]) + "_QTCN_lightning_best.pt")
        if i==0:
            end = time.time()
            print('epoch time',end-start)


    print("------input_size: {:.0f} best test acc is : {:.10f}".format(inputSize, best_test_acc))



if __name__ == "__main__":

    for i in range(0,5):
        run_experiment(device="lightning.qubit",inputSize=5,window_size=2,factor_num=i,epochs=1,lr=0.005,batch_size_train=2,batch_size_test=120,verbose=True)
