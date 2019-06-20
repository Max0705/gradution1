import pandas as pd
import time
directory = './data/chengdudata/'


def load_data():
    weather = pd.read_csv(directory + 'weather.csv')
    # for i in range(1, 31):



if __name__ == '__main__':
    timestamp = 1478366285
    x = time.localtime(timestamp)

    print(x)
