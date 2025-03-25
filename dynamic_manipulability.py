import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from common import bodynames
np.set_printoptions(precision=4, suppress=True)

# this code was used to check how to do DME; it is deprecated

model_id = 2
model_filename = [
    'gimbal.xml',
    '5bar.xml',
    '2linkserial.xml'
][model_id]
actjnt_idxs, act_idxs = [
    [[0, 1], [0, 1]],
    [[2, 4], [0, 1]],
    [[0, 0], [0, 0]]
][model_id]


model = mujoco.MjModel.from_xml_path(f'simple_models/{model_filename}')
#print('a', model.jnt_actfrclimited)
data = mujoco.MjData(model)

#print(*list(zip(bodynames(model.names, model.name_bodyadr), model.body_mass)), sep='\n')

print('--:', model.paths)
print(f'{model.nv = }')

data_i = []
data_j = []
data_if = []
data_jf = []

pos_end_x = []
pos_end_y = []
pos_end_z = []

print(data.qpos)

data.qpos[0] = 0
data.qpos[1] = 0
#'''

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_check = time.time()
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)

        if True: #time.time() - last_check > 1:
            jacp, jacd = np.zeros([3, 2]), np.zeros([3, 2])
            fullm = np.zeros([2, 2])
            #mujoco.mj_kinematics(model, data)
            #mujoco.mj_comPos(model, data)
            mujoco.mj_jac(model, data, jacp, jacd, data.xpos[4], 3)
            mujoco.mj_fullM(model, fullm, data.qM)
            fulljac = np.vstack([jacp, jacd])
            m_jacPI = fullm @ np.linalg.pinv(fulljac)
            DME = m_jacPI.T @ m_jacPI
            val, vec = np.linalg.eig(DME[:3, :3])
            val[val == 0] = 1e-5
            val = np.sqrt(val)
            val = 1 / val
            val /= 10
            print(list(fulljac))
            #print(f'{vec[:, 0]}  {val[0]}  {vec[:, 1]}  {val[1]}  {vec[:, 2]}  {val[2]}')
            model.body_pos[4] = data.xpos[3]
            model.body_pos[5] =   vec[:, 0] * val[0]
            model.body_pos[6] =   vec[:, 1] * val[1]
            model.body_pos[7] =   vec[:, 2] * val[2]
            model.body_pos[8] =  -vec[:, 0] * val[0]
            model.body_pos[9] =  -vec[:, 1] * val[1]
            model.body_pos[10] = -vec[:, 2] * val[2]

            last_check = time.time()

        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            