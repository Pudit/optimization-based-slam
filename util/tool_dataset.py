import numpy as np
import pandas as pd

def read_data(path):
    file = open(path, 'r').readlines()

    data = {}

    for line in file:
        line = line.replace('\n', '')
        line_split = line.split(" ")
        data_type = line_split[0]

        if not(data_type in data):
            data[data_type] = init_type(data_type)

        if data_type in ["VERTEX_SE2", "VERTEX2"]:
            new_data = extract_vertex_se2(line_split[1:])
        elif data_type in ["EDGE_SE2", "EDGE2"]:
            new_data = extract_edge_se2(line_split[1:])

        for key in data[data_type].keys():
            data[data_type][key]+= new_data[key]

    return data

def init_type(data_type):
    print(data_type)
    if data_type in ["VERTEX_SE2", "VERTEX2"]:
        data_dict = init_vertex_se2()
    elif data_type in ["EDGE_SE2", "EDGE2"]:
        data_dict = init_edge_se2()

    return data_dict

# def insert_data(data, new_info):

def init_vertex_se2():
    return {"id": [], "x": [], "y": [], "th": []}

def init_edge_se2():
    return {"pairs": [], "z": [], "info_matrix": []}

def extract_vertex_se2(data):
    # https://www.dropbox.com/s/uwwt3ni7uzdv1j7/g2oVStoro.pdf?dl=0
    # VERTEX_SE2 0 0.000000 0.000000 0.000000 (ID, x, y, th)
    id, x, y, th = data
    data_dict = {"id": [int(id)], "x": [float(x)], "y": [float(y)], "th": [float(th)]}
    return data_dict

def extract_edge_se2(data):
    # https://www.dropbox.com/s/uwwt3ni7uzdv1j7/g2oVStoro.pdf?dl=0
    # IDout IDin dx dy dth I11 I12 I13, I22, I23, I33 for g2o, (but this is not the case for toro)
    IDout, IDin, dx, dy, dth, I11, I12, I13, I22, I23, I33 = data
    pair = np.array([int(IDout), int(IDin)])
    z = np.array([dx, dy, dth], np.float64)
    info_matrix = np.array([[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]], np.float64)
    data_dict = {"pairs": [pair], "z": [z], "info_matrix": [info_matrix]}
    return data_dict


if __name__ == '__main__':
    from visualization import *


    path = "./dataset/input_M3500_g2o.g2o"

    data = read_data(path)
    print(data.keys)
    exit()
    k = "VERTEX_SE2"
    print(data[k].keys())
    for sk in data[k].keys():
        print(sk)
        print(data[k][sk][-1])
        print(np.array(data[k][sk]).shape)
    plot_traj(data[k])
    k = 'EDGE_SE2'
    print(data[k].keys())
    for sk in data[k].keys():
        print(sk)
        print(data[k][sk][-1])
        print(np.array(data[k][sk]).shape)
