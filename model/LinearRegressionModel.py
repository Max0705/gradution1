import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from BasicModule import BasicModule

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.model_name = 'linear'
        self.linear = nn.Linear(43, 1)

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
                               'climate_midraintosmallrain', 'climate_smallraintosunny',
                               'holiday', 'food', 'hotel', 'transport', 'life', 'tourism', 'entertainment', 'sport',
                               'education', 'culture',
                               'hospital', 'shopping', 'car', 'finance', 'house', 'company', 'government', 'entrance',
                               'nature',
                               'counts'])
    for i in range(1, 8):
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df = pd.read_csv('../new_data/data/data_count_poi/order_' + day, index_col=[0])
        data_train = data_train.append(df)
    data_train = data_train.reset_index(drop=True)
    x_train = np.array(data_train.iloc[:, 0:43], dtype=np.float32).reshape(len(data_train), 43)
    y_train = np.array(data_train.iloc[:, 43], dtype=np.float32).reshape(len(data_train), 1)
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
                              'climate_midraintosmallrain', 'climate_smallraintosunny',
                              'holiday', 'food', 'hotel', 'transport', 'life', 'tourism', 'entertainment', 'sport',
                              'education', 'culture',
                              'hospital', 'shopping', 'car', 'finance', 'house', 'company', 'government', 'entrance',
                              'nature',
                              'counts'])
    for i in range(10, 11):
        df = pd.read_csv('../new_data/data/data_count_poi/order_' + str(i), index_col=[0])
        data_test = data_test.append(df)
    data_test = data_test.reset_index(drop=True)
    x_test = np.array(data_test.iloc[:, 0:43], dtype=np.float32).reshape(len(data_test), 43)
    y_test = np.array(data_test.iloc[:, 43], dtype=np.float32).reshape(len(data_test), 1)
    x_test = torch.from_numpy(x_test)

    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-8)

    num_epochs = 1000
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
    out = predict.data.numpy()

    mae = mean_absolute_error(y_test, out)
    mse = mean_squared_error(y_test, out)
    r2 = r2_score(y_test, out)
    print('MAE:' + str(mae))
    print('MSE:' + str(mse))
    print('R2:' + str(r2))
