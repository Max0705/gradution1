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


def divide_map(longitude, latitude):
    num = 0
    # result = []
    for i in range(1, N + 1):
        num += 1
        if longitude < longitude_edge[i]:
            # result.append(longitude_edge[i])
            # result.append(longitude_edge[i - 1])

            break

    for j in range(1, N + 1):

        if latitude < latitude_edge[j]:
            # result.append(latitude_edge[j])
            # result.append(latitude_edge[j - 1])
            break
        num += 100

    # result.append(num)
    return num


itemnames = ['food', 'hotel', 'transport', 'life', 'tourism', 'entertainment', 'sport', 'education', 'culture',
             'hospital', 'shopping', 'car', 'finance', 'house', 'company', 'government', 'entrance', 'nature']

df = pd.DataFrame()
label = []
grid = []
for name in itemnames:
    print(name)
    data = pd.read_csv("../data/poi/" + name + "_poi.csv")
    for item in zip(data['latitude'], data['longitude']):
        grid.append(divide_map(item[1], item[0]))
        label.append(name)

df['label'] = label
df['grid'] = grid

temp = df.groupby(['label', 'grid']).size().rename('counts').reset_index()
temp = temp.sort_values(by=['label', 'grid'])
temp.to_csv("../data/poi/poi_count.csv")
