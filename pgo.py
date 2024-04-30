import numpy as np
# from util_math import *
from util.edge import *
from util.util_math import *
from util.pgo_tool_dataset import *

from scipy.sparse import csr_matrix
from scipy.optimize import least_squares
from scipy.optimize import fsolve
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve, splu
from util.visualization import *
from tqdm import tqdm

class PoseGraph():
    def __init__(self, nodes, edges, n):
        self.nodes = nodes
        self.edges = edges
        self.n = 3


    def init_linear_system(self):
        n = self.n
        no_nodes = len(self.nodes)
        self.H = np.zeros((no_nodes*n,no_nodes*n), dtype=np.float64)
        self.b = np.zeros((no_nodes*n,1), dtype=np.float64)

    def loss(self, nodes):
        """
        Compute loss of graph: sum e_ij.T Omega_ij e_ij
        """
        e = 0.
        # For each constraint
        for edge in self.edges:

            Z_ij = Homo(edge.z)
            omega = edge.info_matrix

            # Get node info
            x_i = nodes[edge.parent]
            x_j = nodes[edge.child]

            X_i = Homo(x_i)
            X_j = Homo(x_j)

            # Calculate error vector
            e_ij = get_pose(np.linalg.inv(Z_ij) @ (np.linalg.inv(X_i) @ X_j)).T

            e += e_ij.T@omega@e_ij

        return e

    def build_linear_system(self):
        """
        Build H matrix and b: H dx = -b
        """

        self.init_linear_system()

        # For each constraint
        for edge in self.edges:

            Z_ij = Homo(edge.z)
            omega = edge.info_matrix

            # Get node info
            x_i = self.nodes[edge.parent]
            x_j = self.nodes[edge.child]

            X_i = Homo(x_i)
            X_j = Homo(x_j)
            R_i = X_i[:2,:2]
            R_z = Rot(edge.z)[:2, :2]
            # R_z = Z_ij[:2,:2]

            s = np.sin(x_i[2])
            c = np.cos(x_i[2])
            # the derivative of R wrt theta
            dR_i = np.array([[-s, -c], [c, -s]], dtype=np.float64)
            dt_ij = np.array([x_j[:2] - x_i[:2]], dtype=np.float64).T


            # Calculate error vector
            e_ij = np.array([get_pose(np.linalg.inv(Z_ij) @ (np.linalg.inv(X_i) @ X_j))]).T

            # Caluclate jacobians
            # NOTE! A =  A_ij, B = B_ij
            # Jacobain
            # A = [-R_z.T R_i.T  | R_z.T dR_i.T dt_ij]
            #   = [ 0 0          | -1                 ]
            A = np.r_[np.c_[-R_z.T @ R_i.T, R_z.T @ dR_i.T @ dt_ij], np.array([[0, 0, -1]])]

            # B = [R_z.T R_i.T |  0]
            #     [  0     0   |  1]
            B = np.r_[np.c_[R_z.T @ R_i.T, np.array([[0], [0]])], np.array([[0, 0, 1]])]

            # Formulate blocks
            H_ii =  A.T @ omega @ A
            H_ij =  A.T @ omega @ B
            H_ji =  B.T @ omega @ A
            H_jj =  B.T @ omega @ B
            b_iT  = e_ij.T @ omega @ A
            b_jT  = e_ij.T @ omega @ B

            # Update H
            n = self.n
            idx_i = edge.parent
            idx_j = edge.child

            self.H[(n * idx_i) : (n * (idx_i + 1)), (n * idx_i) : (n * (idx_i + 1))] += H_ii
            self.H[(n * idx_i) : (n * (idx_i + 1)), (n * idx_j) : (n * (idx_j + 1))] += H_ij
            self.H[(n * idx_j) : (n * (idx_j + 1)), (n * idx_i) : (n * (idx_i + 1))] += H_ji
            self.H[(n * idx_j) : (n * (idx_j + 1)), (n * idx_j) : (n * (idx_j + 1))] += H_jj

            # update b
            self.b[(n * idx_i) : (n * (idx_i + 1))] += b_iT.T
            self.b[(n * idx_j) : (n * (idx_j + 1))] += b_jT.T

        # print(self.H)
        # print(self.b)

    def gn(self, iternum, save_log=False):
        """
        Solve Hx = -b by using gauss-newton
        """
        for i in range(iternum):
            self.build_linear_system()

            self.H[:3,:3] += np.eye(3)

            H_sparse = csr_matrix(self.H) # coo_matrix
            H_sparse = self.H

            dx = spsolve(H_sparse, -self.b)

            dx[:3] = [0,0,0]
            # dx[np.isnan(dx)] = 0
            dpose = np.reshape(dx, (len(self.nodes), 3))
            print(f"loss = {self.loss(self.nodes)}")
            self.nodes += dpose # update
            self.fix_nodes()

            print(self.residual_function(self.nodes.flatten())/10**6)
            if save_log:
                self.save_loss(i)

    def fix_nodes(self):
        """
        Fix the first node at (0, 0) and transform the rest w.r.t. to the first node
        """
        n = self.n
        nodes = self.nodes
        # fixed relative translation
        nodes -= nodes[0]
        th0 = nodes[0, -1]
        R = Rot(nodes[0])[:n-1, :n-1]
        # transform x, y
        self.nodes[:, :n-1] = (R@nodes[:, :n-1].T).T

    def cost(self,x):
        """
        Compute least_squares loss ||Hx+b||^2
        """
        x  = x.flatten()
        x0 = self.nodes.flatten()
        return np.linalg.norm(self.H@(x-x0)+self.b)

    def lm(self, iternum, save_log=False):
        """
        Solving Hx = -b by levenberg marquardt
        """
        lamb = 0.1

        g = lambda x: self.residual_function(x)
        # g = lambda x:l
        for i in tqdm(range(iternum)):
            self.build_linear_system()

            H = self.H
            H[:3, :3] += np.eye(3)
            b = self.b

            LU = splu(H+lamb*np.diag(np.diag(H)), permc_spec="COLAMD")
            dx = LU.solve(-b)

            dpose = np.reshape(dx, (len(self.nodes), 3))
            # dpose -= dpose[0]

            pose_t = self.nodes + dpose
            pose_i = self.nodes

            # if g(pose_t.flatten()) < g(pose_i.flatten()):
            print(f"loss(pose_t) = {self.loss(pose_t)}, loss(pose_i) = {self.loss(pose_i)}")
            if self.loss(pose_t) < self.loss(pose_i):
                print(f"iter {i}: update")
                self.nodes = pose_t
                self.fix_nodes()
                lamb /= 10.
            else:
                print(f"iter {i}: reject")
                lamb *= 10.

            print(f"lambda = {lamb}")
            print(self.residual_function(self.nodes.flatten())/10**6)
            if save_log:
                self.save_loss(i)

    def visualize(self):
        p = self.nodes
        dict_p = {"x": p[:, 0], "y": p[:, 1], "th": p[:, 2]}
        plot_traj(dict_p)


    def residual_function(self, x):
        return np.linalg.norm(self.H@x + self.b)

    def gradient(self):
        lr = self.lr
        return -1/2*lr*np.linalg.inv(self.H)@self.b

    def powell(self, iternum, save_log=False):
        """
        Solving Hx = -b by powell's dog leg
        """
        delta = 1000
        for i in tqdm(range(iternum)):
            pose_graph.build_linear_system()
            H_sparse = csr_matrix(self.H)
            h_gn = spsolve(self.H.T @ self.H, -self.H.T @ self.b)
            g = self.H.T @ self.b
            d_sd = -g
            alpha = (norm(g))**2 / (norm(self.H @ g))**2
            h_sd = alpha * d_sd
            h_sd = h_sd.reshape(h_gn.shape)

            if norm(h_gn) <= delta:
                h_dl = h_gn
            elif norm(h_sd) >= delta:
                h_dl = (delta / norm(h_sd)) * h_sd
            else:
                def radius_fn(beta):
                    return norm(h_sd + beta * (h_gn - h_sd)) - delta
                beta = fsolve(radius_fn, 1)
                h_dl = h_sd + beta[0] * (h_gn - h_sd)

            nodes_new = self.nodes + np.reshape(h_dl, (len(self.nodes), 3))
            if h_dl.all() == h_gn.all() :
                L = norm(self.H @ self.nodes.flatten() + self.b)
            elif h_dl.all() == ((-delta / norm(g)) * g.shape(h_gn.shape)).all():
                L = norm(delta / (2 * alpha) * (2 * norm(alpha * g) - delta))
            else :
                L = norm(1/2 * alpha * (1 - beta**2)* norm(g)**2 + beta * (2 - beta) * (self.H @ self.nodes.flatten() + self.b))

            gain = norm((self.H @ nodes_new.flatten()) - (self.H @ self.nodes.flatten())) / L
            if gain > 0:
                self.nodes = nodes_new
                g = self.H.T @ self.b

            if gain > 0.75:
                delta = max(delta, 2 * np.linalg.norm(h_dl))
            elif gain < 0.25:
                delta = delta / 2

            if save_log:
                self.save_loss(i)
        self.fix_nodes()

    def save_loss(self, iteration):

        if iteration == 0:
            self.loss_list = []
            self.chi_list = []

        self.loss_list.append(self.loss(self.nodes))
        self.chi_list.append(self.chi2())

        with open('./output/loss.npy', 'wb') as f:
            np.save(f, np.array(self.loss_list))
        with open('./output/chi.npy', 'wb') as f:
            np.save(f, np.array(self.chi_list))


    def chi2(self):
        from graphslam.graph import Graph
        import os
        os.remove("./output/output.g2o")
        p = self.nodes
        formatted_data = ""
        for index, values in enumerate(p):
            formatted_data += "VERTEX_SE2 {} {:.17e} {:.17e} {:.17e}\n".format(index, *values)
        formatted_data = formatted_data.strip()
        with open('dataset/M3500_edge_ground_truth.g2o', 'r') as file:
            content = file.read()
        formatted_data += "\n" + content
        with open('./output/output.g2o', 'w') as file:
            file.write(formatted_data)
        g = Graph.from_g2o("./output/output.g2o")
        # print("chi2 = ", g.calc_chi2())
        return g.calc_chi2()

if __name__ == '__main__':
    path_vertex = './dataset/M3500_vertex.g2o'
    path_edge = './dataset/M3500_edge.g2o'

    read_vertex(path_vertex)
    read_edge(path_edge)
    nodes, edges = read_data(path_vertex, path_edge)
    n = 3
    pose_graph = PoseGraph(nodes, edges, n)
    pose_graph.gn(10)
    p = pose_graph.nodes

    dict_p = {"x": p[:, 0], "y": p[:, 1], "th": p[:, 2]}
    plot_traj(dict_p)
    plot_constraints(pose_graph.edges, pose_graph.nodes)
