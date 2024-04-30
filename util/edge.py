import numpy as np

class Edge:
    def __init__(self, id_parent, id_child, measurements, information_matrix):
        self.parent = id_parent
        self.child = id_child
        self.z = measurements
        self.info_matrix = information_matrix

    def get_info(self):
        print(f"parent : {self.parent}")
        print(f"child : {self.child}")
        print(f"measurements : {self.z}")
        print(f"information matrix : {self.info_matrix}")
