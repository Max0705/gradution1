import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 2000)
l = []
for i in range(1, 15):
    if i >= 10:
        day = str(i)
    else:
        day = '0' + str(i)
    df = pd.read_csv('../new_data/data/data_holiday/order_' + day, index_col=[0])
    x = df['grid_num'].tolist()
    l.extend(x)
    l = list(set(l))
    l.sort()

print(len(l))

# for i in range(1, 31):
#     if i >= 10:
#         day = str(i)
#     else:
#         day = '0' + str(i)
#     df = pd.read_csv('../new_data/data/data_classify_poi/order_' + day, index_col=[0])
#     df = df[df['grid_num'].isin(l)]
#     df.to_csv('../new_data/data/data_classify_poi_clean_na/order_' + day,)
#
#
INPUT_SIZE = 7
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
for i in range(1, 15):
    print(i)
    if i >= 10:
        day = str(i)
    else:
        day = '0' + str(i)
    df = pd.read_csv('../new_data/data/data_count_poi_clean_na/order_' + day, index_col=[0])
    data = data.append(df)

data = data.reset_index(drop=True)
all = data['counts'].tolist()
max_value = np.max(all)
min_value = np.min(all)
scalar1 = max_value - min_value
print(scalar1)

total = []

for num in range(len(l)):
    if num % 100 == 0:
        print(num)

    item = l[num]
    d = data[data.grid_num == item].sort_values(by=['month_day', 'hour'])
    m = d['month_day'].tolist()
    h = d['hour'].tolist()
    temp = d['counts']

    lf = []
    temp = temp.tolist()

    temp = list(map(lambda x: x / scalar1, temp))

    for i in range(0, len(temp) - INPUT_SIZE + 1):
        x = temp[i:i + INPUT_SIZE]
        x.append(item)
        x.append(m[i + INPUT_SIZE - 1])
        x.append(h[i + INPUT_SIZE - 1])
        lf.append(x)
    total.extend(lf)
# print(total)
total = pd.DataFrame(total,
                     columns=['past6', 'past5', 'past4', 'past3', 'past2', 'past1', 'counts_r', 'grid_num', 'month_day',
                              'hour'])

result = pd.merge(total, data, on=['grid_num', 'month_day', 'hour'], how='left')

order = ['past6', 'past5', 'past4', 'past3', 'past2', 'past1', 'counts_r', 'month_day', 'week', 'hour', 'temperature',
         'humidity',
         'wind_grade',
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
         'counts']
result = result[order]
print(result)
result.to_csv('../new_data/data/total.csv')
