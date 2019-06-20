import pandas as pd
import time

directory = '../data/chengdudata/'

day = "01"
data = pd.read_csv(directory + 'chengdu_201611' + day + '/order_201611' + day,
                   names=['id', 'in_time', 'out_time', 'in_longitude', 'in_latitude', 'out_longitude',
                          'out_latitude'])

x1 = time.mktime(time.strptime("2016/11/01 12:00", "%Y/%m/%d %H:%M"))

x2 = time.mktime(time.strptime("2016/11/01 14:00", "%Y/%m/%d %H:%M"))

temp = data[data.in_time > x1].reset_index(drop=True)
temp = temp[temp.in_time < x2].reset_index(drop=True)

x = temp['in_longitude'].tolist()
y = temp['in_latitude'].tolist()

print(x)
print(y)

with open("../data/view_gps/data.js", "w") as f:
    f.write("var data = {\"data\":[")
    for i in range(0, len(temp)):
        f.write("[" + str(temp.iloc[i]['in_longitude']) + "," + str(temp.iloc[i]['in_latitude']) + "],\n")

    f.write("]}\n")