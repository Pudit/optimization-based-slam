import numpy as np
from util.edge import *
from util.visualization import *

def read_data(path_vertex, path_edge):
    nodes = read_vertex(path_vertex)
    edges = read_edge(path_edge)
    return nodes, edges

def read_vertex(path):
    # skip the VERTEX stuff
    vertex_data = np.loadtxt(path, usecols=range(1, 5))
    nodes = vertex_data[:, 1:]

    return nodes

def read_edge(path):
    # skip the EDGE stuff
    edge_data = np.loadtxt(path, usecols=range(1, 12))

    edges = []

    for di in edge_data:
        # assume g2o format !!! add others soon !!!
        ei = information_matrix_g2o(di)
        edges.append(ei)

    return edges

def information_matrix_g2o(data):
    # https://www.dropbox.com/s/uwwt3ni7uzdv1j7/g2oVStoro.pdf?dl=0
    # IDout IDin dx dy dth I11 I12 I22 I33 I13 I23
    # g2o format
    IDout, IDin, dx, dy, dth, I11, I12, I13, I22, I23, I33 = data
    info_matrix = np.array([[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]], np.float64)
    z = np.array([dx, dy, dth], np.float64)

    # id_parent, id_child, measurements, information_matrix
    e = Edge(int(IDout), int(IDin), z, info_matrix)

    return e

if __name__ == '__main__':
    path_vertex = './dataset/M3500_vertex.g2o'
    path_edge = './dataset/M3500_edge.g2o'

    read_vertex(path_vertex)
    read_edge(path_edge)
    nodes, edges = read_data(path_vertex, path_edge)
    print(nodes[0])
    p = nodes
    dict_p = {"x": p[:, 0], "y": p[:, 1], "th": p[:, 2]}
    plot_traj(dict_p)
