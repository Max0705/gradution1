import pandas as pd

data = pd.DataFrame(columns=
                    ['month_day', 'week', 'hour', 'temperature', 'humidity', 'wind_grade', 'grid_max_longitude',
                     'grid_min_longitude',
                     'grid_max_latitude', 'grid_min_latitude', 'grid_num', 'climate_sunny', 'climate_rain',
                     'climate_smallrain',
                     'climate_overcast', 'climate_cloudy', 'climate_overcasttocluody',
                     'climate_overcasttosmallrain',
                     'climate_overcasttosunny', 'climate_cloudytosunny', 'climate_cloudytoovercast',
                     'climate_cloudytosmallrain',
                     'climate_midraintosmallrain', 'climate_smallraintosunny', 'counts'])
for i in range(1, 31):
    if i >= 10:
        day = str(i)
    else:
        day = '0' + str(i)
    df = pd.read_csv('../data/data_onehot/order_' + day, index_col=[0])
    data = data.append(df)
data = data.reset_index(drop=True)

temp1 = len(data[data.counts <= 5])
print(str(temp1))

temp2 = len(data[(data.counts <= 20) & (data.counts > 5)])
print(str(temp2))

temp3 = len(data[(data.counts <= 50) & (data.counts > 20)])
print(str(temp3))
temp4 = len(data[(data.counts <= 100) & (data.counts > 50)])
print(str(temp4))
temp5 = len(data[data.counts > 100])
print(str(temp5))

# temp1 = data.groupby(['grid_num'])['counts'].sum().reset_index().sort_values(by=['counts'])
# print(temp1)
# pd.set_option('max_column',100)
# print(data[data.grid_num == 5463])
