import time
import pandas as pd

directory = '../data/poi/'

itemnames = ['food', 'hotel', 'life', 'shopping', 'company']

data = pd.DataFrame(columns=['name', 'latitude', 'longitude', 'address', 'region', 'type'])

num = 0
for item in itemnames:
    num += 1
    print(item)
    temp = pd.read_csv(directory + item + '_poi.csv', index_col=[0])
    with open("../data/view_gps/data" + str(num) + ".csv", "w") as f:
        for i in range(0, len(temp)):
            if str(temp.iloc[i]['longitude']) == 'nan':
                print("xxxxxxx")
            else:
                f.write(str(temp.iloc[i]['longitude']) + "," + str(temp.iloc[i]['latitude']) + "\n")

