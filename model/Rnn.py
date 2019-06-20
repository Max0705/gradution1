import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Rnn, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state


INPUT_SIZE = 43
TIME_STEP = 10
learning_rate = 0.02
num_epochs = 10
batch_size = 64

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
        df = pd.read_csv('../new_data/data/data_count_poi_clean_na/order_' + day, index_col=[0])
        data_train = data_train.append(df)
    l = data_train['grid_num'].drop_duplicates().tolist()
    data_train = data_train.sort_values(by=['grid_num', 'month_day', 'hour'])
    data_train = data_train.reset_index(drop=True)
    x_train = np.array(data_train.iloc[:, 0:43], dtype=np.float32).reshape(len(l), 168, 43)

    print(x_train)
    y_train = np.array(data_train.iloc[:, 43], dtype=np.float32).reshape(len(l), 168, 1)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    # train_dataset = TensorDataset(x_train, y_train)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # print(train_loader)

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
    for i in range(8, 9):
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df = pd.read_csv('../new_data/data/data_count_poi_clean_na/order_' + day, index_col=[0])
        data_test = data_test.append(df)
    l = data_test['grid_num'].drop_duplicates().tolist()
    x_test = np.array(data_test.iloc[:, 0:43], dtype=np.float32).reshape(len(l), 24, 43)
    data_test = data_test.reset_index(drop=True)
    y_test = np.array(data_test.iloc[:, 43], dtype=np.float32).reshape(len(l), 24, 1)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    # test_dataset = TensorDataset(x_test, y_test)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print(x_test.shape)
    print(y_test.shape)
    model = Rnn(INPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    h_state = None

    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        target = Variable(y_train)

        out, h_state = model(inputs, h_state)
        # h_state = Variable(h_state.data)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 1000 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

    model.eval()

    eval_loss = 0
    eval_acc = 0

    test_inputs = Variable(x_test)
    test_target = Variable(y_test)

    out, h_state = model(test_inputs, h_state)
    loss = criterion(out, test_target)

    print(out)
    print(y_test)
    print(loss.item())

    y_test = y_test.data.numpy()
    out = out.data.numpy()

    sum = 0
    for i in range(len(y_test)):
        sum += math.fabs(y_test[0][0][i] - out[0][0][i]) / len(y_test)

    print(sum)



    # eval_loss += loss.data.item() * target.size(0)
    # _, pred = torch.max(out, 1)
    # num_correct = (pred == target).sum()
    # eval_acc += num_correct.item()
    #
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    #     eval_loss / (len(test_dataset)),
    #     eval_acc / (len(test_dataset))
    # ))
