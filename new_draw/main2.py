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
    pd.set_option('max_column', 100)
    data = pd.read_csv('../new_data/data/data_classify_poi_clean_na/order_24', index_col=[0])

    data = data[data.hour == 14]
    data = data[['grid_max_longitude', "grid_min_longitude", "grid_max_latitude", "grid_min_latitude", "classify"]]
    print(data[data.classify == 0])

    plt.figure(figsize=(26, 15))
    # plt.xlim(104.045, 104.097)
    # plt.ylim(30.648, 30.678)
    plt.xlim(103.259070, 104.589952)
    plt.ylim(30.17542, 31.09167)
    osmhandler = OSMHandler()
    osmhandler.apply_file('map2.osm')
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

    color = ['#FFD2D2', '#FF9797', '#FF5151', '#FF0000', '#AE0000', '#750000', '#4D0000', '#2F0000']
    i = 0

    for item in zip(data['grid_min_longitude'], data["grid_max_longitude"], data["grid_min_latitude"],
                    data["grid_max_latitude"], data['classify']):
        # print(i)
        i = i + 1
        if i != 1080 & i != 0:
            # print(i)
            # print(item)
            x = [item[0], item[1], item[1], item[0]]
            y = [item[2], item[2], item[3], item[3]]
            plt.xlim(103.259070, 104.589952)
            plt.ylim(30.17542, 31.09167)
            plt.fill(x, y, facecolor=color[item[4]], alpha=0.8)

    plt.show()
    # plt.plot(x, y, "r*")
    plt.axis('off')


if __name__ == '__main__':
    # main()
    print('Done!')

    color = ['#FFD2D2', '#FF9797', '#FF5151', '#FF0000', '#AE0000', '#750000', '#4D0000', '#2F0000']
    plt.figure(figsize=(80, 10))
    plt.xlim(0, 8)
    plt.ylim(0, 1)

    plt.fill([0, 1, 1, 0], [0, 0, 1, 1], facecolor=color[0], alpha=0.8)
    plt.fill([1, 2, 2, 1], [0, 0, 1, 1], facecolor=color[1], alpha=0.8)
    plt.fill([2, 3, 3, 2], [0, 0, 1, 1], facecolor=color[2], alpha=0.8)
    plt.fill([3, 4, 4, 3], [0, 0, 1, 1], facecolor=color[3], alpha=0.8)
    plt.fill([4, 5, 5, 4], [0, 0, 1, 1], facecolor=color[4], alpha=0.8)
    plt.fill([5, 6, 6, 5], [0, 0, 1, 1], facecolor=color[5], alpha=0.8)
    plt.fill([6, 7, 7, 6], [0, 0, 1, 1], facecolor=color[6], alpha=0.8)
    plt.fill([7, 8, 8, 7], [0, 0, 1, 1], facecolor=color[7], alpha=0.8)
    plt.show()
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
