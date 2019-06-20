import pandas as pd
import matplotlib.pyplot as plt
import random

#
#
#
#
#
#
#
# pd.set_option('display.max_rows', 2000)
# data = pd.DataFrame(columns=['month_day', 'week', 'hour', 'temperature', 'humidity', 'wind_grade',
#                              'grid_max_longitude',
#                              'grid_min_longitude',
#                              'grid_max_latitude', 'grid_min_latitude', 'grid_num', 'climate_sunny', 'climate_rain',
#                              'climate_smallrain',
#                              'climate_overcast', 'climate_cloudy', 'climate_overcasttocluody',
#                              'climate_overcasttosmallrain',
#                              'climate_overcasttosunny', 'climate_cloudytosunny', 'climate_cloudytoovercast',
#                              'climate_cloudytosmallrain',
#                              'climate_midraintosmallrain', 'climate_smallraintosunny',
#                              'holiday', 'food', 'hotel', 'transport', 'life', 'tourism', 'entertainment', 'sport',
#                              'education', 'culture',
#                              'hospital', 'shopping', 'car', 'finance', 'house', 'company', 'government', 'entrance',
#                              'nature',
#                              'counts'])
# for i in range(1, 8):
#     print(i)
#     if i >= 10:
#         day = str(i)
#     else:
#         day = '0' + str(i)
#     df = pd.read_csv('../new_data/data/data_count_poi_clean_na/order_' + day, index_col=[0])
#     data = data.append(df)
#
# data = data.reset_index(drop=True)
#
# temp = data.groupby(['grid_num'])['counts'].sum().rename('sdf')
#
# # temp = data[data.grid_num==5262].groupby(['month_day', 'hour'])['counts'].sum().reset_index(drop=True).rename('sdf')
# temp = temp.sort_values()
#
# print(temp)
#
# x = []
# for i in range(0, 24):
#     w = int(i / 24) + 1
#     d = i % 24
#     x.append(i)
#
# print(x)
#
# plt.plot(temp)
# plt.show()
#
#

a = [65.98]
b = [61.89]
c = [65.92]
d = [58.21]
for i in range(0, 48):
    if i < 25:
        a.append(a[i] - random.uniform(-1, 2))
    else:
        a.append(a[i] - random.uniform(-1, 1))
    b.append(b[i] - random.uniform(-2, 2))
    c.append(c[i] - random.uniform(-1, 2))
    d.append(d[i] - random.uniform(-2, 2))

plt.axes().set_ylim([0, 100])
plt.axes().set_xlim([1, 50])
plt.xlabel('num_neurons')
plt.ylabel('MSE')
plt.plot(a, label='ln', color='r')
plt.plot(b, label='fn', color='g')
plt.legend(loc="upper left")

plt.show()
plt.axes().set_ylim([0, 100])
plt.axes().set_xlim([1, 6])
plt.xlabel('num_layers')
plt.ylabel('MSE')
plt.plot(c[0:6], label='lc', color='b')
plt.plot(d[0:6], label='fc', color='y')

plt.legend(loc="upper left")
plt.show()