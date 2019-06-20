import math

EARTH_REDIUS = 6378.137


def rad(d):
    return d * math.pi / 180.0


def get_distance(graph, e1, e2):
    lon1, lat1 = graph.node[e1]['pos']
    lon2, lat2 = graph.node[e2]['pos']

    radLat1 = rad(float(lat1))
    radLat2 = rad(float(lat2))
    a = radLat1 - radLat2
    b = rad(float(lon1)) - rad(float(lon2))
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2)
                     + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s * 1000


def get_distance_(lon1, lat1, lon2, lat2):
    radLat1 = rad(float(lat1))
    radLat2 = rad(float(lat2))
    a = radLat1 - radLat2
    b = rad(float(lon1)) - rad(float(lon2))
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2)
                                + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s * 1000
