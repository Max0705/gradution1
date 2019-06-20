import json

import osmium as osm
import networkx as nx

from distance import get_distance


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
    # g1 = {Id: i+1 for i, Id in enumerate(sorted(osmhandler.G.nodes(), key=int))}
    # g2 = {i+1: Id for i, Id in enumerate(sorted(osmhandler.G.nodes(), key=int))}
    with open('./out/Graph.txt', 'w', encoding='utf-8') as f:
        f.write('c XXX graph\n')
        f.write(f'p edge {len(osmhandler.G.nodes())} {len(osmhandler.G.edges())}\n')
        # for node_id in range(1, len(g1)+1):
        #     f.write(f'n {node_id} {osmhandler.G.node[g2[node_id]]["pos"][0]} {osmhandler.G.node[g2[node_id]]["pos"][1]}\n')
        for node in osmhandler.G.nodes():
            f.write(f'n {node} {osmhandler.G.node[node]["pos"][0]} {osmhandler.G.node[node]["pos"][1]}\n')
        for edge in osmhandler.G.edges():
            e1, e2 = edge
            f.write(f'e {e1} {e2} {get_distance(osmhandler.G, e1, e2)}\n')


def main():
    osmhandler = OSMHandler()
    osmhandler.apply_file('./data/map_highway.osm')
    write_to_file(osmhandler)


if __name__ == '__main__':
    main()
    print('Done!')
