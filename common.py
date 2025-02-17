import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def bodynames(allnames, bodyidxs):
    allnames = allnames.decode()
    return [[i, allnames[bodyidxs[i]:allnames.find('\x00', bodyidxs[i])]] for i in range(len(bodyidxs))]


def normalize(arr):
    arr -= arr.min()
    arr /= arr.max()
    return arr

def converge_constraint(m, d):
    ''' Should update only constraints? does NOT account for gravity for sure '''
    d.qacc *= 0
    i = 0
    while ((max(abs(d.qacc)) > 1e-3 or max(abs(d.qvel)) > 1e-3) and i < 5000) or i == 0:
        i += 1
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_makeConstraint(m, d)
        mujoco.mj_transmission(m, d)
        mujoco.mj_projectConstraint(m, d)
        mujoco.mj_fwdVelocity(m, d)
        mujoco.mj_fwdActuation(m, d)
        mujoco.mj_fwdAcceleration(m, d)
        mujoco.mj_fwdConstraint(m, d)
        mujoco.mj_Euler(m, d)
    if max(abs(d.qacc)) > 1e-3 or max(abs(d.qvel)) > 1e-3:
        return False
        '''print('converge_constraint error')
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                viewer.sync()'''
    return True

def DME(m, d, pos, body, compute=False, jacobian=False):
    # DME https://drive.google.com/file/d/12f1hfklP6O7x7lS7g7alw3GfwRk5ctH7/view?usp=sharing
    # and there is other metric MFE https://drive.google.com/file/d/18Lba3GnN9YxJdj9zpmfm-_6KdTSdh_gL/view?usp=sharing
    # (second not implemented)
    jacp, jacd = np.zeros([3, m.nv]), np.zeros([3, m.nv])
    fullm = np.zeros([m.nv, m.nv])
    if compute:
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
    mujoco.mj_jac(m, d, jacp, jacd, pos, body)
    mujoco.mj_fullM(m, fullm, d.qM)
    fulljac = np.vstack([jacp, jacd])
    if jacobian:
        return fulljac
    m_jacPI = fullm @ np.linalg.pinv(fulljac)
    DME = m_jacPI.T @ m_jacPI
    val, vec = np.linalg.eig(DME[:3, :3])
    val[val == 0] = 1e-5
    val = np.sqrt(val)
    val = 1 / val
    val /= 10

    return val, vec.T