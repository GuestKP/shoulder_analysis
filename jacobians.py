from common import names, get_rotmat
import numpy as np
import mujoco


def J_5bar_get_idxs(m):
    joint_names = names(m, 'joint')
    return [
        [
            joint_names.index(pre+post)
            for pre in ['j_', 'j_prim2sec_']
        ]
        for post in ['a', 'b']
    ]

def J_5bar(d, jidxs, endidx, endaxis=np.array([0, -1, 0])):
    v = get_rotmat(d, endidx) @ endaxis
    J = np.zeros([2, 3])
    for i, [ui, wi] in enumerate(jidxs):
        u = d.xaxis[ui]
        w = d.xaxis[wi]
        #print(v, u, w, np.cross(w, v), np.dot(np.cross(u, w), v))
        J[i] = np.cross(w, v) / np.dot(np.cross(u, w), v)
    return J.T


def J_gimbal(m, d):
    jacp, jacd = np.zeros([3, m.nv]), np.zeros([3, m.nv])
    fullm = np.zeros([m.nv, m.nv])
    mujoco.mj_jac(m, d, jacp, jacd, d.xpos[-1], 2)
    mujoco.mj_fullM(m, fullm, d.qM)
    return np.vstack([jacp, jacd])