import pandas as pd
import matplotlib.pyplot as plt

f = open('result2.txt', 'r')
x = f.readline()
x = x.strip().split(' ')
x = [float(i) for i in x]
print(len(x))

data = pd.DataFrame()
for i in range(8, 15):
    if i >= 10:
        day = str(i)
    else:
        day = '0' + str(i)
    if i == 8:
        data = pd.read_csv('../new_data/data/data_count_poi_clean_na/order_' + day, index_col=[0])
    else:
        t = pd.read_csv('../new_data/data/data_count_poi_clean_na/order_' + day, index_col=[0])
        data = data.append(t)

data = data.reset_index(drop=True)
tem = data.groupby(['grid_num'])['counts'].sum().sort_values()
print(tem)
data = data[data.grid_num == 5262]['counts']
y = data.tolist()
ix = data.index.tolist()

x = [x[i] for i in ix]
print(x)

plt.plot(x)
plt.plot(y)

plt.show()
