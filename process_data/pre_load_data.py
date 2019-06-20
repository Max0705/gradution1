import pandas as pd
import time

directory = './data/chengdudata/'


def pre_load():
    weather = pd.read_csv(directory + 'weather.csv')
    weather_day = []
    weather_hour = []
    climate_new = []
    for item in weather['time']:
        x = time.mktime(time.strptime(item, "%Y/%m/%d %H:%M"))
        x = time.localtime(x)
        weather_day.append(x.tm_mday)
        weather_hour.append(x.tm_hour)
    weather['day'] = weather_day
    weather['hour'] = weather_hour
    weather = weather.drop(['city', 'time', 'wind_direction'], axis=1)

    # data_with_weather = pd.DataFrame(
    #     columns=['id', 'in_time', 'out_time', 'in_longitude', 'in_latitude', 'out_longitude', 'out_latitude'])
    all_missing = []
    for i in range(1, 31):
        t1 = time.time()
        print(str(i))
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        data = pd.read_csv(directory + 'chengdu_201611' + day + '/order_201611' + day,
                           names=['id', 'in_time', 'out_time', 'in_longitude', 'in_latitude', 'out_longitude',
                                  'out_latitude'])
        # data_with_weather = data_with_weather.append(temp)

        data = data.drop(['id', 'out_time', 'out_longitude', 'out_latitude'], axis=1)

        month_day = []
        week = []
        hour = []
        minute = []
        temperature = []
        humidity = []
        wind_grade = []
        climate = []
        grid_max_longitude = []
        grid_min_longitude = []
        grid_max_latitude = []
        grid_min_latitude = []
        grid_num = []

        i = 0
        for item in zip(data['in_time'], data['in_longitude'], data['in_latitude']):
            if i % 100000 == 0:
                print(i)

            i += 1
            # item = data_with_weather.iloc[i]['in_time']
            temp = int(item[0])
            temp = time.localtime(temp)
            month_day.append(temp.tm_mday)
            week.append(temp.tm_wday)
            hour.append(temp.tm_hour)

            minute.append(int(temp.tm_min / 10))
            try:
                temp_weather = get_weather(weather, temp.tm_mday, temp.tm_hour)
            except:
                temp_weather = [0, 0, 0, 0]
                record = str(temp.tm_mday) + 'd' + str(temp.tm_hour) + 'h'
                if record not in all_missing:
                    all_missing.append(record)

            temperature.append(temp_weather[0])
            humidity.append(temp_weather[1])
            wind_grade.append(temp_weather[2])
            climate.append(temp_weather[3])

            temp_grid = divide_map(item[1], item[2])
            grid_max_longitude.append(temp_grid[0])
            grid_min_longitude.append(temp_grid[1])
            grid_max_latitude.append(temp_grid[2])
            grid_min_latitude.append(temp_grid[3])
            grid_num.append(temp_grid[4])

        data['month_day'] = month_day
        data['week'] = week
        data['hour'] = hour
        # data['minute'] = minute
        data['temperature'] = temperature
        data['humidity'] = humidity
        data['wind_grade'] = wind_grade
        data['climate'] = climate
        data['grid_max_longitude'] = grid_max_longitude
        data['grid_min_longitude'] = grid_min_longitude
        data['grid_max_latitude'] = grid_max_latitude
        data['grid_min_latitude'] = grid_min_latitude
        data['grid_num'] = grid_num

        data.drop(['in_time'], axis=1)
        data.to_csv('./data/data_with_weather/order_' + day)
        print('time cost：' + str(time.time() - t1))
    print('missing data_with_weather:' + str(all_missing))
    return 1


def get_weather(weather_data, day, hour):
    result = []

    temp = weather_data[(weather_data.day == day) & (weather_data.hour == hour)]

    result = [int(temp['temperature'].values[0]), int(temp['humidity'].values[0]),
              int(temp['wind_grade'].values[0]),
              str(temp['climate'].values[0])]

    return result


# divide the map2.osm.osm to N*N grids
# map2.osm.osm information
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
    result = []
    for i in range(1, N + 1):
        num += 1
        if longitude < longitude_edge[i]:
            result.append(longitude_edge[i])
            result.append(longitude_edge[i - 1])

            break

    for j in range(1, N + 1):
        if latitude < latitude_edge[j]:
            result.append(latitude_edge[j])
            result.append(latitude_edge[j - 1])
            break
        num += 100

    result.append(num)
    return result


# count orders per hour per grid
def count():
    for i in range(1, 31):
        print(i)
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        data = pd.read_csv('./data/data_with_weather/order_' + day, index_col=[0])
        # temp = data.groupby(['hour', 'minute', 'grid_num']).size().rename('counts').reset_index()
        # temp = pd.merge(data, temp, on=['hour', 'minute', 'grid_num'], how='left')
        # temp = temp.drop(['in_time', 'in_longitude', 'in_latitude'], axis=1)
        # temp = temp.sort_values(by=['hour', 'minute', 'grid_num'])
        temp = data.groupby(['hour', 'grid_num']).size().rename('counts').reset_index()
        temp = pd.merge(data, temp, on=['hour', 'grid_num'], how='left')
        temp = temp.drop(['in_time', 'in_longitude', 'in_latitude'], axis=1)
        temp = temp.sort_values(by=['hour', 'grid_num'])
        temp = temp.drop_duplicates().reset_index(drop=True)
        temp.to_csv('data/data_count/order_' + day)
    return 1


all_climate = ['晴', '雨', '小雨', '阴', '多云', '阴转多云', '阴转小雨', '阴转晴', '多云转晴', '多云转阴', '多云转小雨', '中雨转小雨', '小雨转晴']


# transform climate to one-hot encoding
def climate_to_onehot():
    for i in range(1, 31):
        print(i)
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        data = pd.read_csv('./data/data_count/order_' + day, index_col=[0])
        climate_sunny = []
        climate_rain = []
        climate_smallrain = []
        climate_overcast = []
        climate_cloudy = []
        climate_overcasttocluody = []
        climate_overcasttosmallrain = []
        climate_overcasttosunny = []
        climate_cloudytosunny = []
        climate_cloudytoovercast = []
        climate_cloudytosmallrain = []
        climate_midraintosmallrain = []
        climate_smallraintosunny = []
        for item in data['climate']:
            climate_sunny.append(judge_climate(item, 0))
            climate_rain.append(judge_climate(item, 1))
            climate_smallrain.append(judge_climate(item, 2))
            climate_overcast.append(judge_climate(item, 3))
            climate_cloudy.append(judge_climate(item, 4))
            climate_overcasttocluody.append(judge_climate(item, 5))
            climate_overcasttosmallrain.append(judge_climate(item, 6))
            climate_overcasttosunny.append(judge_climate(item, 7))
            climate_cloudytosunny.append(judge_climate(item, 8))
            climate_cloudytoovercast.append(judge_climate(item, 9))
            climate_cloudytosmallrain.append(judge_climate(item, 10))
            climate_midraintosmallrain.append(judge_climate(item, 11))
            climate_smallraintosunny.append(judge_climate(item, 12))

        data = data.drop(['climate'], axis=True)
        data['climate_sunny'] = climate_sunny
        data['climate_rain'] = climate_rain
        data['climate_smallrain'] = climate_smallrain
        data['climate_overcast'] = climate_overcast
        data['climate_cloudy'] = climate_cloudy
        data['climate_overcasttocluody'] = climate_overcasttocluody
        data['climate_overcasttosmallrain'] = climate_overcasttosmallrain
        data['climate_overcasttosunny'] = climate_overcasttosunny
        data['climate_cloudytosunny'] = climate_cloudytosunny
        data['climate_cloudytoovercast'] = climate_cloudytoovercast
        data['climate_cloudytosmallrain'] = climate_cloudytosmallrain
        data['climate_midraintosmallrain'] = climate_midraintosmallrain
        data['climate_smallraintosunny'] = climate_smallraintosunny

        temp = data['counts']
        data = data.drop(['counts'], axis=1)
        data['counts'] = temp
        data.to_csv('./data/data_onehot/order_' + day)
    return 1


def judge_climate(item, n):
    if str(item) == all_climate[n]:
        return 1
    else:
        return 0


def classify_counts():
    for i in range(1, 31):
        print(i)
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        data = pd.read_csv('../new_data/data/data_count_poi/order_' + day, index_col=[0])

        # 1:1 2-5:2 5-20:3 20-50:4 50-100:5 100+:6
        classify = []
        for item in data['counts']:
            count = int(item)
            if count == 0:
                classify.append(0)
            if count == 1:
                classify.append(1)
            if (count > 1) & (count <= 5):
                classify.append(2)
            if (count > 5) & (count <= 20):
                classify.append(3)
            if (count > 20) & (count <= 50):
                classify.append(4)
            if (count > 50) & (count <= 100):
                classify.append(5)
            if count > 100:
                classify.append(6)

        data = data.drop(['counts'], axis=1)
        data['classify'] = classify
        data.to_csv('../new_data/data/data_classify_poi/order_' + day)


def load_poi(thread):
    itemnames = ['food', 'hotel', 'transport', 'life', 'tourism', 'entertainment', 'sport', 'education', 'culture',
                 'hospital', 'shopping', 'car', 'finance', 'house', 'company', 'government', 'entrance', 'nature']
    data = pd.read_csv('../data/poi/poi_count.csv', index_col=[0])
    for i in range(thread, thread+1):
        print(i)
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df1 = pd.read_csv('../data/data_fillna/order_' + day, index_col=[0])

        for name in itemnames:
            print(name)
            temp = []
            for item in df1['grid_num']:
                x = data[(data.label == name) & (data.grid == item)]['counts']
                if len(x) != 0:
                    temp.append(x.item())
                else:
                    temp.append(0)
            df1[name] = temp

        temp = df1.counts
        df1 = df1.drop(['counts'], axis=1)
        df1['counts'] = temp
        df1.to_csv('../data/data_count_poi/order_' + day)


def holiday():
    for i in range(1, 31):
        print(i)
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df1 = pd.read_csv('../data/data_onehot/order_' + day, index_col=[0])

        holiday = []

        for item in df1['week']:

            if (item == 5) | (item == 6):
                holiday.append(1)
            else:
                holiday.append(0)

        df1['holiday'] = holiday
        temp = df1.counts
        df1 = df1.drop(['counts'], axis=1)
        df1['counts'] = temp

        df1.to_csv('../data/data_holiday/order_' + day)


def fill_na():
    for i in range(1, 31):
        print(i)
        if i >= 10:
            day = str(i)
        else:
            day = '0' + str(i)
        df1 = pd.read_csv('../data/data_holiday/order_' + day, index_col=[0])

        for hour in range(0, 24):
            for grid in range(1, 10001):

                if grid % 1000 == 0:
                    print(hour)
                    print(grid)

                temp = df1[(df1.hour == hour) & (df1.grid_num == grid)]
                if len(temp) == 0:
                    other = df1[df1.hour == hour]
                    info = [longitude_edge[grid % 100], longitude_edge[grid % 100 - 1],
                            latitude_edge[int(grid / 100)], latitude_edge[int(grid / 100) - 1], grid]
                    new_row = [int(df1.month_day[0]), int(df1.week[0]), hour]
                    weather1 = other.iloc[0, 3:6].values.tolist()
                    weather2 = other.iloc[0, 11:25].values.tolist()
                    new_row.extend(weather1)
                    new_row.extend(info)
                    new_row.extend(weather2)
                    new_row.append(0)

                    df1.loc[len(df1)] = new_row
                    print(df1)

        df1 = df1.sort_values(by=['month_day', 'week', 'hour', 'grid_num'])
        print(df1)
        df1.to_csv('../data/data_fillna/order_' + day)


if __name__ == '__main__':
    # pre_load()
    # pd.set_option('max_column', 100)
    # count()
    # climate_to_onehot()
    classify_counts()
    # holiday()
    # fill_na()
    # load_poi(1)
