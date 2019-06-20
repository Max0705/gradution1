import os
import json
import collections
import random

import networkx as nx

import func
from distance import get_distance_
from EdgeData import EdgeData


def main():
    graph = nx.Graph()
    visited = {}  # 已经访问过的点
    edge_dict = collections.defaultdict(EdgeData)
    traj = open('./out/traj.txt', 'w')

    # 先统计一天24个时段
    # i为天数，j为时段数
    for i in range(1, 2):
        for j in range(1, 2):
            base_dir = func.dir_index(i, j)
            for name in os.listdir(base_dir):
                # print(base_dir + '/' + name)
                try:
                    one_data = func.get_one_gpx_data(base_dir + '/' + name)
                except:
                    continue
                for k in range(len(one_data)-1):
                    lon, lat = one_data[k]
                    next_lon, next_lat = one_data[k+1]
                    if (lon, lat) not in visited:
                        visited[(lon, lat)] = len(graph.nodes) + 1
                        graph.add_node(len(graph.nodes) + 1, pos=(lon, lat))
                    if (next_lon, next_lat) not in visited:
                        visited[(next_lon, next_lat)] = len(graph.nodes) + 1
                        graph.add_node(len(graph.nodes) + 1, pos=(next_lon, next_lat))
                    graph.add_edge(visited[(lon, lat)], visited[(next_lon, next_lat)],
                                   length=get_distance_(lon, lat, next_lon, next_lat))
                for lon, lat in one_data:
                    traj.write(f'{visited[(lon, lat)]} ')
                traj.write('\n')
                for k in range(len(one_data)-1):
                    index = random.randint(j * 2 - 2, j * 2 - 1)
                    lon, lat = one_data[k]
                    next_lon, next_lat = one_data[k + 1]
                    n1 = visited[(lon, lat)]
                    n2 = visited[(next_lon, next_lat)]
                    edge_str = func.edge_str(n1, n2)  # 路段的字符串
                    speed = graph[n1][n2]['length'] / (random.randint(5, 8))
                    edge_dict[edge_str].slots[index].add_data(speed)
    func.save_dimacs(graph)
    traj.close()

    # 轨迹信息
    a = {}
    for edge_str, edge_data in edge_dict.items():
        a[edge_str] = {}
        for i in range(48):
            slot = edge_data.slots[i]
            a[edge_str][i + 1] = [slot.flow, slot.v]
    with open('./out/edge.json', 'w') as f:
        json.dump(a, f)


if __name__ == '__main__':
    main()
    # print('Done')
