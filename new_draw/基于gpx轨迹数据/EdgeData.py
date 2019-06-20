# import func

# edge_dict = {
#     # 点到点
#     '1->2': {
#         # slot: [流量，平均速度]
#         '1': [10, 6],
#         '2': [12, 7],
#         # ...
#         '48': [10, 8]
#     }
# }


class Slot(object):
    def __init__(self):
        self.flow = 0
        self.v = .0  # 平均速度，初始化为0

    def add_data(self, v):
        """增加一辆车经过，v为其速度"""
        self.v = (self.v * self.flow + v) / (self.flow + 1)
        self.flow += 1


class EdgeData(object):
    def __init__(self):
        # self.edge_str = edge_str
        self.slots = [Slot() for _ in range(48)]
