import networkx as nx
import matplotlib.pyplot as plt


def easy_draw(graph):
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos=pos, node_size=0)
    plt.show()


def draw_graph(graph, save=False):
    pos = nx.get_node_attributes(graph, 'pos')
    node_color = nx.get_node_attributes(graph, 'cluster_id')

    nx.draw_networkx_edges(graph, pos=pos, alpha=0.4)
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=list(node_color.keys()), 
                           node_size=0.1, node_color=list(node_color.values()))

    if save:
        plt.savefig('./out/graph.png', dpi=400)
    else:
        plt.show()


def draw_graph_with_paths(graph, *paths):
    """画出多条路径的简易图"""
    for path, color in paths:
        x, y = list(), list()
        for node in path:
            x.append(graph.node[node]['pos'][0])
            y.append(graph.node[node]['pos'][1])
        plt.plot(x, y, color=color)
    # plt.savefig('./out/paths.png', dpi=400)
    plt.show()


def limit_lon_lat():
    plt.xlim(118.066, 118.197)
    plt.ylim(24.424, 24.561)


if __name__ == '__main__':
    pass
