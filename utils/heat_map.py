import pandas as pd

N = 100
max_longitude = 104.589952
min_longitude = 103.259070
max_latitude = 31.09167
min_latitude = 30.17542
longitude_unit = (max_longitude - min_longitude) / N
latitude_unit = (max_latitude - min_latitude) / N
longitude_edge = [min_longitude + longitude_unit * i for i in range(0, N + 1)]
latitude_edge = [min_latitude + latitude_unit * i for i in range(0, N + 1)]

data = pd.DataFrame(columns=['month_day', 'week', 'hour', 'temperature', 'humidity', 'wind_grade', 'climate',
                             'grid_max_longitude', 'grid_min_longitude', 'grid_max_latitude', 'grid_min_latitude',
                             'grid_num', 'counts'])
for i in range(1, 31):
    print(i)
    if i >= 10:
        day = str(i)
    else:
        day = '0' + str(i)

    df = pd.read_csv('../data/data_count/order_' + day, index_col=[0])
    data = data.append(df)

data = data.reset_index(drop=True)
temp = data[['grid_num', 'counts']].groupby(['grid_num'])['counts'].sum().rename('total').reset_index()


def get_gps(grid):
    lon = []
    lat = []
    for grid_num in grid:
        grid_num = grid_num - 201
        x = grid_num % 100
        y = int(grid_num / 100)
        t_lon = (longitude_edge[x] + longitude_edge[x - 1]) / 2
        t_lat = (latitude_edge[y] + latitude_edge[y + 1]) / 2
        lon.append(t_lon)
        lat.append(t_lat)

    return lon, lat


data1 = pd.DataFrame()
data1['grid_num'] = [i for i in range(202, 10157)]
data1['lon'], data1['lat'] = get_gps(data1['grid_num'])
data1 = pd.merge(data1, temp, on=['grid_num'], how='right')
print(data1)
with open("../data/heatmap/heat_map_data.txt", "w") as f:
    f.write("[")
    for i in range(0, len(data1)):
        f.write("{\"lng\":" + str(data1.iloc[i]['lon']) + ", \"lat\":" + str(data1.iloc[i]['lat']) + ", \"count\":" + str(
            data1.iloc[i]['total']) + "},\n")

    f.write("]")
