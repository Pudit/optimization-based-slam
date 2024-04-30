import numpy as np
from util.util_math import *
from util.visualization import *

class StochasticGradientDecent:
    def __init__(self, iteration = 100):
        self.cost = 0
        self.gamma = np.array([1e12, 1e12, 1e12])
        self.iteration = iteration

    def init_M(self, numstates):
        self.M = np.zeros((numstates, 3))

    def update_M(self):
        # Update an approximation matrix M
        pairs = self.pairs
        p = self.p
        # for each constraint (loop closure)
        for idx, pair in enumerate(pairs):
            a, b = pair
            R = Rot(p[a])
            # get sigma^-1
            cov = np.linalg.inv(self.info_matrix[idx])

            W = np.linalg.inv(R@cov@R.T)
            diag_W = np.diag(W)

            # for all i in a+1 to b update Mi
            self.M[a+1:b+1, :] += diag_W
            # Update gamma
            for i in range(3):
                self.gamma[i] = np.min([self.gamma[i], diag_W[i]])

    def modified_sgd(self, iteration):
        pairs = self.pairs
        p = self.p
        t = self.t
        info_matrix = self.info_matrix
        gamma = self.gamma
        M = self.M
        N = M.shape[0]

        for idx, pair in enumerate(pairs):

            a, b = pair
            R = Rot(p[a])
            tab = t[idx]
            Tab = Homo(tab)
            Pa = Homo(p[a])
            Pb = Pa@Tab
            p_b = get_pose(Pb)
            r = p_b-p[b]
            r[2] = mod2pi(r[2])
            cov_ab = np.linalg.inv(self.info_matrix[idx])
            d = 2*np.linalg.inv(R.T@cov_ab@R)@r

            # update x, y, and theta
            # vectorize version
            alpha = 1/(iteration*gamma)
            total_w = np.sum(1/M[a+1:b+1], axis = 0)
            beta = (b-a)*d*alpha

            # for j in range(3):
            #     if np.abs(beta[j]) > np.abs(r[j]):
            #         beta[j] = r[j]
            #     dpose = 0
            #     for i in range (a, N):
            #         if a + 1 <= i <= b + 1:
            #             dpose = dpose + beta[j] / M[i][j] / total_w[j]
            #         p[i][j] = p[i][j] + dpose

            # vectorize
            # shape(a+1:b+1) = (1, 3)
            filter = np.abs(beta) > np.abs(r)
            beta[filter] = r[filter]

            # compute incremental poses
            M_partial = M[a+1:b+1]
            d_pose = beta/M_partial
            d_pose = d_pose/total_w
            d_pose = np.cumsum(d_pose, axis=0)

            # Update poses
            p[a+1:b+1] += d_pose
            p[b+1:] += d_pose[-1]

        self.p = p

    def optimize(self, states, measurements, save_log=False):
        # get poses
        self.p = np.column_stack((states["x"], states["y"], states["th"]))
        # get measurements
        self.t = measurements["z"]
        # get loop closure constraints
        self.pairs = measurements["pairs"]
        # get infomation matrix
        self.info_matrix = measurements["info_matrix"]

        numstates = self.p.shape[0]
        self.init_M(numstates)
        if save_log:
            loss_list = []
            chi_list = []
        for iteration in range(1, self.iteration+1):
            print("iteration = ", iteration)
            self.update_M()
            self.modified_sgd(iteration)

            if save_log:
                loss_list.append(self.loss())
                chi_list.append(self.compute_chi())

        # visualization
        p = self.p
        dict_p = {"x": p[:, 0], "y": p[:, 1], "th": p[:, 2]}
        plot_traj(dict_p)

        if save_log:
            with open('./output/loss.npy', 'wb') as f:
                np.save(f, np.array(loss_list))
            with open('./output/chi.npy', 'wb') as f:
                np.save(f, np.array(chi_list))

    def loss(self):
        # compute loss
        pairs = self.pairs
        p = self.p
        t = self.t
        info_matrix = self.info_matrix

        e = 0
        for idx, pair in enumerate(pairs):
            a, b = pair
            Z_ij = Homo(t[idx])
            omega = info_matrix[idx]
            x_i = p[a]
            x_j = p[b]
            X_i = Homo(x_i)
            X_j = Homo(x_j)
            e_ij = get_pose(np.linalg.inv(Z_ij) @ (np.linalg.inv(X_i) @ X_j)).T
            e += e_ij.T @ omega @ e_ij

        return e

    def compute_chi(self):
        from graphslam.graph import Graph
        import os
        os.remove("./output/output.g2o")
        # get poses
        p = self.p

        # get measurements
        t = self.t
        # get loop closure constraints
        pairs = self.pairs
        # # get infomation matrix
        # self.info_matrix

        formatted_data = ""

        for index, values in enumerate(p):
            formatted_data += "VERTEX_SE2 {} {:.6e} {:.6e} {:.6e}\n".format(index, *values)

        # for index, pair in enumerate(pairs):
        #     im = self.info_matrix[index]
        #     I11 = im[0, 0]
        #     I12 = im[0, 1]
        #     I13 = im[0, 2]
        #     I22 = im[1, 1]
        #     I23 = im[1, 2]
        #     I33 = im[2, 2]
        #
        #     dx, dy, dth = self.t[index]
        #
        #     IDout, IDin = pair
        #
        #     values = [IDout, IDin, dx, dy, dth, I11, I12, I13, I22, I23, I33]
        #
        #     formatted_data += f"EDGE_SE2 {IDout} {IDout} {dx} {dy} {dth} {I11} {I12} {I13} {I22} {I23} {I33}\n"

        formatted_data = formatted_data.strip()

        with open('dataset/M3500_edge_ground_truth.g2o', 'r') as file:
            content = file.read()
        formatted_data += "\n" + content
        with open('./output/output.g2o', 'w') as file:
            file.write(formatted_data)
        g = Graph.from_g2o("./output/output.g2o")

        return g.calc_chi2()


if __name__ == '__main__':
    from util.tool_dataset import *

    path = "./dataset/input_M3500_g2o.g2o"

    data = read_data(path)
    data_keys = data.keys()
    states_type = ["VERTEX_SE2", "VERTEX2"]
    for k in states_type:
        if k in data_keys:
            break
    states = data[k]

    print(f"type(states['x']): {type(states['x'])}")

    mea_keys = ["EDGE2", "EDGE_SE2"]
    for k in mea_keys:
        if k in data_keys:
            break
    measurements = data[k]
    print(type(measurements))
    SGD = StochasticGradientDecent(iteration=100)

    SGD.optimize(states, measurements)
