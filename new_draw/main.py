import json

import osmium as osm
import networkx as nx

from distance import get_distance
from draw_demo import draw_graph, easy_draw
import matplotlib.pyplot as plt
import pandas as pd
import time

N = 100
max_longitude = 104.589952
min_longitude = 103.259070
max_latitude = 31.09167
min_latitude = 30.17542
longitude_unit = (max_longitude - min_longitude) / N
latitude_unit = (max_latitude - min_latitude) / N
longitude_edge = [min_longitude + longitude_unit * i for i in range(0, N + 1)]
latitude_edge = [min_latitude + latitude_unit * i for i in range(0, N + 1)]

class OSMHandler(osm.SimpleHandler):

    def __init__(self):
        super(OSMHandler, self).__init__()
        self.G = nx.Graph()

    def node(self, n):
        lon, lat = str(n.location).split('/')
        self.G.add_node(n.id, pos=(lon, lat))

    def way(self, w):
        for i, n in enumerate(w.nodes):
            if i != len(w.nodes) - 1:
                a, b = n.ref, w.nodes[i + 1].ref
                self.G.add_edge(a, b)

    def relation(self, r):
        pass


def write_to_file(osmhandler):
    g1 = {Id: i + 1 for i, Id in enumerate(sorted(osmhandler.G.nodes(), key=int))}
    g2 = {i + 1: Id for i, Id in enumerate(sorted(osmhandler.G.nodes(), key=int))}
    with open('./out/ChengDU_Graph.txt', 'w', encoding='utf-8') as f:
        f.write('c ChengDu graph\n')
        f.write(f'p edge {len(osmhandler.G.nodes())} {len(osmhandler.G.edges())}\n')
        for node_id in range(1, len(g1) + 1):
            f.write(
                f'n {node_id} {osmhandler.G.node[g2[node_id]]["pos"][0]} {osmhandler.G.node[g2[node_id]]["pos"][1]}\n')
        for edge in osmhandler.G.edges():
            e1, e2 = edge
            f.write(f'e {g1[e1]} {g1[e2]} {get_distance(osmhandler.G, e1, e2)}\n')


def main():
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

    plt.figure(figsize=(26, 15))
    # plt.xlim(104.045, 104.097)
    # plt.ylim(30.648, 30.678)
    plt.xlim(103.259070, 104.589952)
    plt.ylim(30.17542, 31.09167)
    osmhandler = OSMHandler()
    osmhandler.apply_file('map.osm')
    write_to_file(osmhandler)
    graph = osmhandler.G
    pos = nx.get_node_attributes(graph, 'pos')
    node_color = nx.get_node_attributes(graph, 'cluster_id')

    nx.draw_networkx_edges(graph, pos=pos, alpha=0.4)
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=list(node_color.keys()),
                           node_size=0.1, node_color=list(node_color.values()))

    # color = ['r*', 'c*', 'y*', 'm*', 'b*']
    #
    # for i in range(1, 6):
    #     data = pd.read_csv("../data/view_gps/data" + str(i) + ".csv", sep=',', names=['x', 'y'])
    #     x = data['x'].tolist()
    #     y = data['y'].tolist()
    #     plt.plot(x, y, color[i-1])

    for i in range(N):
        plt.plot(longitude_edge)
    plt.plot(x, y, "r*")
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
    print('Done!')
# http://overpass-api.de/api/map?bbox=103.259070,30.17542,104.589952,31.09167
# http://overpass-api.de/api/map?bbox=104.045,30.653,104.097,30.673

# max_longitude = 104.589952
# min_longitude = 103.259070
# max_latitude = 31.09167
# min_latitude = 30.17542

# 104.046675,30.673821000000004
# 104.047215,30.652887
# 104.097809,30.67347
# 104.09787,30.652654
# 104.045,30.653,30.673, 104.097
#
# 0.052,0.02
