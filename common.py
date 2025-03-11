import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation


def names(m, type='body'):
    allnames = m.names.decode()
    idxs = []
    if type == 'body':
        idxs = m.name_bodyadr
    if type == 'joint':
        idxs = m.name_jntadr
    if type == 'equality':
        idxs = m.name_eqadr
    if type == 'actuator':
        idxs = m.name_actuatoradr
    return [allnames[idxs[i]:allnames.find('\x00', idxs[i])] for i in range(len(idxs))]


def find_jnt_and_act(m, target_names=['j_a', 'j_b']):
    #joints = names(m, 'joint')
    act_names = names(m, 'actuator')
    tacts = [act_names.index(tn) for tn in target_names]
    tjnts = [int(m.actuator_trnid[it][0]) for it in tacts]
    return tjnts, tacts


def is_constrained(m):
    return m.name_eqadr.shape[0] > 0


def get_endeffector_idx(m, target_name='l_endeffector'):
    return names(m, 'body').index(target_name)


def get_endeffector_pos(d, idx):
    return d.xpos[idx]

def get_rotmat(d, idx):
    return Rotation.from_quat(d.xquat[idx]).as_matrix()

def get_euler(d, idx):
    return Rotation.from_quat(d.xquat[idx]).as_euler('XYZ')


def normalize(arr):
    arr -= arr.min()
    arr /= arr.max()
    return arr

def converge_constraint(m, d, v=None):
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
        if v:
            if v.is_running():
                v.sync()
                time.sleep(m.opt.timestep)
    if max(abs(d.qacc)) > 1e-3 or max(abs(d.qvel)) > 1e-3:
        return False
    return True

def DME(m, d, pos, body, compute=False, jacobian=False, axis=True):
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
    if axis:
        val, vec = np.linalg.eig(DME[3:, 3:])
    else:
        val, vec = np.linalg.eig(DME[:3, :3])
    val[val == 0] = 1e-5
    val = np.sqrt(val)
    val = 1 / val
    val /= 10

    return val, vec.T