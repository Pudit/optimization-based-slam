import numpy as np

def Rot(p):
    """
    p = state (x, y, th)
    """
    # print(p)
    _, _, th = p
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Homo(p):
    x, y, th = p
    H = Rot(p)
    H[0, 2] = x
    H[1, 2] = y
    return H

def get_pose(P):
    """
    P = homogenous matrix
    """
    x = P[0][2]
    y = P[1][2] 
    th = np.arctan2(P[1][0],P[0][0])
    return np.array([x, y, th])

def mod2pi(r):
    r = (r + np.pi) % (2 * np.pi) - np.pi
    return r

if __name__ == '__main__':
    p = np.array([1, 2, 0])
    print(Rot(p))
    print(Homo(p))
    print(get_pose(Homo(p)))
    print(mod2pi(np.pi*3/2))
