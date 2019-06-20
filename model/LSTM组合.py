import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sp = 286254
le = 583110
GRID_NUM = 1767
TIME_STEP = 6
learning_rate = 0.02
num_epochs = 200
SEQ_LENGTH = 162


class RNN(torch.nn.Module):
    def __init__(self, ln, lc):
        super(RNN, self).__init__()

        self.lstm = nn.ModuleList([torch.nn.LSTM(input_size=TIME_STEP, hidden_size=ln, num_layers=lc) for i in
                                   range(GRID_NUM)])
        self.reg = nn.ModuleList([nn.Linear(ln, 1) for i in range(GRID_NUM)])

    def forward(self, x):
        tem_result = torch.empty(0)
        for i in range(GRID_NUM):
            out, _ = self.lstm[i](x[i])
            out = self.reg[i](out)
            out = out.view(x.shape[2])
            tem_result = torch.cat((tem_result, out), dim=0)
        return tem_result


if __name__ == '__main__':
    pd.set_option('display.max_columns', 2000)
    pd.set_option('display.max_rows', 2000)
    data = pd.read_csv("../new_data/data/total.csv", index_col=[0])

    data1 = data[data.month_day <= 7]
    data2 = data[data.month_day > 7]

    x_train = np.array(data1.iloc[:, 0:6], dtype=np.float32).reshape(GRID_NUM, 1, SEQ_LENGTH, TIME_STEP)
    y_train = np.array(data1.iloc[:, 6], dtype=np.float32).reshape(sp)
    x_test = np.array(data2.iloc[:, 0:6], dtype=np.float32).reshape(GRID_NUM, 1, 330 - SEQ_LENGTH, TIME_STEP)
    y_test = np.array(data2.iloc[:, 6], dtype=np.float32).reshape(le - sp)

    print(x_train)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    model = RNN(16, 12)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    h_state = None
    losspi = []

    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        target = Variable(y_train)

        out = model(inputs)

        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch % 100 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))
        losspi.append(loss.item())

    model.eval()

    eval_loss = 0
    eval_acc = 0

    # x_test = x_train
    # y_test = y_train
    test_inputs = Variable(x_test)
    test_target = Variable(y_test)

    out = model(test_inputs)
    loss = criterion(out, test_target)

    print(out)
    print(loss.item())

    y_test = test_target.data.numpy()
    out = out.data.numpy()

    # out = out.reshape(len(y_test))
    print(y_test)
    print(len(y_test))

    scalar = 1398.0
    out = list(map(lambda x: x * scalar, out))
    y_test = list(map(lambda x: x * scalar, y_test))

    mae = mean_absolute_error(y_test, out)
    mse = mean_squared_error(y_test, out)
    r2 = r2_score(y_test, out)
    print('MAE:' + str(mae))
    print('MSE:' + str(mse))
    print('R2:' + str(r2))

    sum = 0
    for i in range(len(y_test)):
        sum += math.fabs(y_test[i] - out[i]) / len(y_test)

    print(sum)

    # out = out[len(y_test)-100:len(y_test)]
    # y_test = y_test[len(y_test)-100:len(y_test)]
    plt.plot(out, 'r')
    # plt.show()
    plt.plot(y_test)
    plt.show()
    plt.plot(losspi)
    plt.show()

    # eval_loss += loss.data.item() * target.size(0)
    # _, pred = torch.max(out, 1)
    # num_correct = (pred == target).sum()
    # eval_acc += num_correct.item()
    #
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    #     eval_loss / (len(test_dataset)),
    #     eval_acc / (len(test_dataset))
    # ))
