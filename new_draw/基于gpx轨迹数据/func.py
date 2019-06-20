import json
import xml.dom.minidom


# 各路径和映射
begin_dir = '../gpx数据/2014-07-'
mid_dir = '/2014-07-'
end_dir = '.txt-outresult'


def edge_str(n1, n2) -> str:
    return str(n1) + '->' + str(n2)


def _f(i: int) -> str:
    """是否在i前面加上0，视情况而定"""
    if i < 10:
        return '0' + str(i)
    else:
        return str(i)


def dir_index(i: int, j: int) -> str:
    """第i天第j个文件夹的目录"""
    x = _f(j) if i == 1 else str(j)
    return begin_dir + _f(i) + mid_dir + _f(i) + '-' + x + end_dir


def get_one_gpx_data(dir_name: str) -> list:
    """返回此gpx文件的点的数据"""
    dom = xml.dom.minidom.parse(dir_name)
    trkpt = dom.getElementsByTagName('trkpt')
    res = []
    for t in trkpt:
        lat = t.getAttribute('lat')
        lon = t.getAttribute('lon')
        res.append((lon, lat))
    return res


def save_dimacs(graph):
    with open('./out/graph.txt', 'w', encoding='utf-8') as f:
        f.write('p XiaMen Graph\n')
        for n in graph.nodes:
            f.write('n {} {} {}\n'.format(n, graph.node[n]['pos'][0], graph.node[n]['pos'][1]))
        for e in graph.edges:
            n1, n2 = e
            f.write('e {} {} {} ""\n'.format(n1, n2, graph[n1][n2]['length']))
