import pandas as pd
from urllib.request import urlopen, quote
import json
import time
import numpy as np
#
# max_longitude = 104.589952
# min_longitude = 103.259070
# max_latitude = 31.09167
# min_latitude = 30.17542
#
# ak = 'qBxiHzAnrv916iOjHpLxs0ll5aIGiDGl'
# query = "美食"
# query = quote(query)
# url = "http://api.map.baidu.com/place/v2/search?query="+query+"&bounds=30.17542,103.259070,31.09167,104.589952&output=json&ak=" + ak
#
# req = urlopen(url)
# res = req.read().decode()
# temp = json.loads(res)
#
# with open("poi.json", "w") as f:
#         f.write(json.dumps(temp, indent=4, ensure_ascii=False)+'\n')


import torch

# x = torch.randn(2038, 168, 32)
# print(x)
# print(x.shape)
# x = x[:, -1, :]
# print(x)
# print(x.shape)

# step = 0
# start, end = step * np.pi, (step+1)*np.pi
#
# steps = np.linspace(start, end, 10, dtype=np.float32)
# x_np = np.sin(steps)  # float32 for converting torch FloatTensor
# y_np = np.cos(steps)
#
# print(x_np[np.newaxis, :, np.newaxis])# shape (batch, time_step, input_size)
# print(y_np[np.newaxis, :, np.newaxis])

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# data_train = pd.DataFrame(columns=
#                           ['month_day', 'week', 'hour', 'temperature', 'humidity', 'wind_grade',
#                            'grid_max_longitude',
#                            'grid_min_longitude',
#                            'grid_max_latitude', 'grid_min_latitude', 'grid_num', 'climate_sunny', 'climate_rain',
#                            'climate_smallrain',
#                            'climate_overcast', 'climate_cloudy', 'climate_overcasttocluody',
#                            'climate_overcasttosmallrain',
#                            'climate_overcasttosunny', 'climate_cloudytosunny', 'climate_cloudytoovercast',
#                            'climate_cloudytosmallrain',
#                            'climate_midraintosmallrain', 'climate_smallraintosunny',
#                            'holiday', 'food', 'hotel', 'transport', 'life', 'tourism', 'entertainment', 'sport',
#                            'education', 'culture',
#                            'hospital', 'shopping', 'car', 'finance', 'house', 'company', 'government', 'entrance',
#                            'nature',
#                            'counts'])
# for i in range(1, 31):
#     if i >= 10:
#         day = str(i)
#     else:
#         day = '0' + str(i)
#     df = pd.read_csv('../data/data_count_poi/order_' + day, index_col=[0])
#     data_train = data_train.append(df)
# data_train = data_train.reset_index(drop=True)
# test = data_train[['month_day', 'week', 'hour', 'grid_num', 'counts']]
# # x = test.groupby(['grid_num']).size()
# pd.set_option('max_row', 10000)
# print(test[test.grid_num == 5455])
#
# x = np.linspace(1, 720, 720)
# print(x)
# for i in range(5460, 6000):
#     y = []
#     temp = test[test.grid_num == i]
#     if len(temp) == 0:
#         continue
#     print(i)
#     for day in range(1, 31):
#         for hour in range(0, 24):
#             t = temp[(temp.month_day == day) & (temp.hour == hour)]['counts']
#             if len(t) == 0:
#                 y.append(0)
#             else:
#                 y.append(int(t))
#     print(y)
#     print(len(y))
#     plt.plot(x, y)
#     plt.show()

x = [[1, 2, 3, 4, 5], [3, 5, 7, 5, 3], [4, 3, 2, 4, 1], [1, 3, 4, 5, 6],
     [1, 2, 3, 4, 5], [3, 5, 7, 5, 3], [4, 3, 2, 4, 1], [1, 3, 4, 5, 6],
     [1, 2, 3, 4, 5], [3, 5, 7, 5, 3], [4, 3, 2, 4, 1], [1, 3, 4, 5, 6],
     [1, 2, 3, 4, 5], [3, 5, 7, 5, 3], [4, 3, 2, 4, 1], [1, 3, 4, 5, 6]]
data = pd.DataFrame()
x_train = np.array(x, dtype=np.float32).reshape(4, 1, 4, 5)
print(x_train)
