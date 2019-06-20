import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# from BasicModule import BasicModule

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.model_name = 'linear2'
        self.linear = nn.Linear(24, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


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

    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-8)

    num_epochs = 100000
    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        target = Variable(y_train)

        out = model(inputs)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

    model.eval()
    predict = model(Variable(x_test))
    predict = predict.data.numpy()


    def result(x):
        if x <= 1.5:
            return 1
        elif x <= 2.5:
            return 2
        elif x <= 3.5:
            return 3
        elif x <= 4.5:
            return 4
        elif x <= 5.5:
            return 5
        else:
            return 6


    predict2 = []
    for item in predict:
        item = int(item)
        item = result(item)
        predict2.append(item)

    print(predict2)
    sum = 0
    for i in range(0, 73709):
        if y_test[i] == predict2[i]:
            sum += 1

    print(sum / 73709)
