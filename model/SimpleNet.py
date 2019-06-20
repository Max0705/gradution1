import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd


class SimpleNet(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, out_dim):
        super(SimpleNet, self).__init__()
        self.model_name = 'SimpleNet'
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


batch_size = 64
learning_rate = 0.02
num_epochs = 10

if __name__ == '__main__':
    data_train = pd.DataFrame(columns=
                              ['month_day', 'week', 'hour', 'temperature', 'humidity', 'wind_grade',
                               'grid_max_longitude',
                               'grid_min_longitude',
                               'grid_max_latitude', 'grid_min_latitude', 'grid_num', 'climate_sunny', 'climate_rain',
                               'climate_smallrain',
                               'climate_overcast', 'climate_cloudy', 'climate_overcasttocluody',
                               'climate_overcasttosmallrain',
                               'climate_overcasttosunny', 'climate_cloudytosunny', 'climate_cloudytoovercast',
                               'climate_cloudytosmallrain',
                               'climate_midraintosmallrain', 'climate_smallraintosunny', 'classify'])
    for i in range(1, 24):
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df = pd.read_csv('../data/data_classify/order_' + day, index_col=[0])
        data_train = data_train.append(df)
    data_train = data_train.reset_index(drop=True)
    x_train = np.array(data_train.iloc[:, 0:24], dtype=np.float32).reshape(239765, 24)
    y_train = np.array(data_train.iloc[:, 24], dtype=np.float32).reshape(239765, 1)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    data_test = pd.DataFrame(columns=
                             ['month_day', 'week', 'hour', 'temperature', 'humidity', 'wind_grade',
                              'grid_max_longitude',
                              'grid_min_longitude',
                              'grid_max_latitude', 'grid_min_latitude', 'grid_num', 'climate_sunny', 'climate_rain',
                              'climate_smallrain',
                              'climate_overcast', 'climate_cloudy', 'climate_overcasttocluody',
                              'climate_overcasttosmallrain',
                              'climate_overcasttosunny', 'climate_cloudytosunny', 'climate_cloudytoovercast',
                              'climate_cloudytosmallrain',
                              'climate_midraintosmallrain', 'climate_smallraintosunny', 'classify'])
    for i in range(24, 31):
        df = pd.read_csv('../data/data_classify/order_' + str(i), index_col=[0])
        data_test = data_test.append(df)
    data_test = data_test.reset_index(drop=True)
    x_test = np.array(data_test.iloc[:, 0:24], dtype=np.float32).reshape(73709, 24)
    y_test = np.array(data_test.iloc[:, 24], dtype=np.float32).reshape(73709, 1)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    print(x_test)
    print(x_test[0])
    model = SimpleNet(24, 30, 10, 6)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        target = Variable(y_train)

        out = model(inputs)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (epoch + 1) % 100 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

    model.eval()

    # eval_loss = 0
    # eval_acc = 0
    # for i in range(0, 73709):
    #     inputs = Variable(x_test[i])
    #     target = Variable(y_test[i])
    #
    #     out = model(inputs)
    #     loss = criterion(out, target)
    #     eval_loss += loss.data.item() * target.size(0)
    #     _, pred = torch.max(F.softmax(out), 1)
    #
    #     num_correct = (pred == target + 1).sum()
    #     eval_acc += num_correct.item()
    #
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    #     eval_loss / (len(x_test)),
    #     eval_acc / (len(x_test))
    # ))

    _, pred = torch.max(F.softmax(out), 1)
    tt = pred.data.numpy()
    np.set_printoptions(threshold=10000000)
    print(tt)
    sum = 0
    for i in range(0, 73709):
        if y_test[i] == tt[i] + 1:
            sum += 1

    print(y_test)
    print(sum / 73709)
