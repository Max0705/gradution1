import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # input应该为(batch_size,seq_len,input_szie)
        x, _ = self.lstm(x)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.out(x)
        x = x.view(s, b, -1)
        return x


INPUT_SIZE = 6
TIME_STEP = 10
learning_rate = 0.02
num_epochs = 2000
batch_size = 64

if __name__ == '__main__':
    pd.set_option('display.max_rows', 2000)
    data = pd.DataFrame(columns=['month_day', 'week', 'hour', 'temperature', 'humidity', 'wind_grade',
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
        print(i)
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df = pd.read_csv('../new_data/data/data_count_poi_clean_na/order_' + day, index_col=[0])
        data = data.append(df)

    data = data.reset_index(drop=True)

    temp = data[data.grid_num == 4267].groupby(['month_day', 'hour'])['counts'].sum().reset_index(drop=True).rename(
        'sdf')

    l = []
    temp = temp.tolist()
    max_value = np.max(temp)
    min_value = np.min(temp)
    scalar1 = max_value - min_value
    temp = list(map(lambda x: x / scalar1, temp))

    for i in range(0, len(temp) - INPUT_SIZE):
        l.append(temp[i:i + INPUT_SIZE])

    x_train = np.array(l, dtype=np.float32).reshape(len(l), 1, INPUT_SIZE)

    y_train = np.array(temp[INPUT_SIZE:len(temp)], dtype=np.float32).reshape(len(l), 1, 1)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    data2 = pd.DataFrame(columns=['month_day', 'week', 'hour', 'temperature', 'humidity', 'wind_grade',
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
                                  'hospital', 'shopping', 'car', 'finance', 'house', 'company', 'government',
                                  'entrance',
                                  'nature',
                                  'counts'])
    for i in range(8, 15):
        print(i)
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df = pd.read_csv('../new_data/data/data_count_poi_clean_na/order_' + day, index_col=[0])
        data2 = data2.append(df)

    data2 = data2.reset_index(drop=True)

    temp = data2[data2.grid_num == 4267].groupby(['month_day', 'hour'])['counts'].sum().reset_index(drop=True).rename(
        'sdf')

    print(temp)
    l = []
    temp = temp.tolist()
    max_value = np.max(temp)
    min_value = np.min(temp)
    scalar2 = max_value - min_value
    temp = list(map(lambda x: x / scalar2, temp))
    for i in range(0, len(temp) - INPUT_SIZE):
        l.append(temp[i:i + INPUT_SIZE])
    print(l)

    x_test = np.array(l, dtype=np.float32).reshape(len(l), 1, INPUT_SIZE)

    y_test = np.array(temp[INPUT_SIZE:len(temp)], dtype=np.float32).reshape(len(l))

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    model = RNN(INPUT_SIZE, 32, 2)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    h_state = None

    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        target = Variable(y_train)

        out = model(inputs)

        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

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

    y_test = y_test.data.numpy()
    out = out.data.numpy()

    out = out.reshape(len(y_test))

    sum = 0
    for i in range(len(y_test)):
        sum += math.fabs(y_test[i] - out[i]) / len(y_test)

    print(sum)
    out = list(map(lambda x: x * scalar2, out))
    y_test = list(map(lambda x: x * scalar2, y_test))
    plt.plot(out, 'r')
    plt.plot(y_test)

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
