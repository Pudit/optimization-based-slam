import numpy as np
import matplotlib.pyplot as plt

def plot_traj(state):
    """
    state = (x,y,th)
    """
    x = state['x']
    y = state['y']
    plt.plot(x, y)
    plt.show()

def plot_constraints(edges, nodes):
    p = nodes
    dict_p = {"x": p[:, 0], "y": p[:, 1]}
    x = dict_p['x']
    y = dict_p['y']
    plt.plot(x, y)
    # print(x[:2])
    # print(y[:2])

    for e in edges:
        # print(e.parent)
        # print(e.child)
        x_p = nodes[e.parent][:2]
        x_c = nodes[e.child][:2]
        if np.abs(e.parent-e.child) == 1:
            continue
        x = [x_p[0], x_c[0]]
        y = [x_p[1], x_c[1]]

        plt.plot(x, y, c='r', linewidth=0.2)
        # plt.show()
        # exit()
    plt.show()

if __name__ == '__main__':
    from pgo_tool_dataset import *

    path_vertex = './dataset/M3500_vertex.g2o'
    path_edge = './dataset/M3500_edge.g2o'
    #
    path_vertex = './dataset/intel_vertex.g2o'
    path_edge = './dataset/intel_edge.g2o'

    # path_vertex = './dataset/M3500a_vertex.g2o'
    # path_edge = './dataset/M3500a_edge.g2o'

    read_vertex(path_vertex)
    read_edge(path_edge)
    nodes, edges = read_data(path_vertex, path_edge)
    # n = 3
    # pose_graph = PoseGraph(nodes, edges, n)
    # pose_graph.lm(40)
    # p = pose_graph.nodes
    # print(p[:10])
    # dict_p = {"x": p[:, 0], "y": p[:, 1], "th": p[:, 2]}
    # plot_traj(dict_p)
    plot_constraints(edges, nodes)
