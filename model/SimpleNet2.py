import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
num_epochs = 1000

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
                               'classify'])
    for i in range(1, 8):
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df = pd.read_csv('../new_data/data/data_classify_poi_clean_na/order_' + day, index_col=[0])
        data_train = data_train.append(df)
    data_train = data_train.reset_index(drop=True)
    x_train = np.array(data_train.iloc[:, 0:43], dtype=np.float32).reshape(len(data_train), 43)
    y_train = np.array(data_train.iloc[:, 43], dtype=np.long).reshape(len(data_train))
    print(x_train)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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
                              'classify'])
    for i in range(8, 15):
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df = pd.read_csv('../new_data/data/data_classify_poi_clean_na/order_' + day, index_col=[0])
        data_test = data_test.append(df)
    data_test = data_test.reset_index(drop=True)
    x_test = np.array(data_test.iloc[:, 0:43], dtype=np.float32).reshape(len(data_test), 43)
    y_test = np.array(data_test.iloc[:, 43], dtype=np.long).reshape(len(data_test))

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet(43, 100, 30, 8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        target = Variable(y_train)

        out = model(inputs)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 1000 == 0:
        print('num:{}, Epoch[{}/{}], loss:{:.6f}'.format(i, epoch + 1, num_epochs, loss.item()))

    model.eval()

    eval_loss = 0
    eval_acc = 0

    inputs = Variable(x_test)
    target = Variable(y_test)

    out = model(inputs)
    loss = criterion(out, target)
    eval_loss += loss.data.item()
    _, pred = torch.max(out, 1)
    num_correct = (pred == target).sum()
    eval_acc += num_correct.item()

    print(out.data.numpy())
    print(out.data.numpy().shape)
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss,
        eval_acc / (len(test_dataset))
    ))

    plt.plot(pred.data.numpy())
    plt.show()
