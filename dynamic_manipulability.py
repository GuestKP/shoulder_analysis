import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
np.set_printoptions(precision=4, suppress=True)

def bodynames(allnames, bodyidxs):
    allnames = allnames.decode()
    return [[i, allnames[bodyidxs[i]:allnames.find('\x00', bodyidxs[i])]] for i in range(len(bodyidxs))]

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


model = mujoco.MjModel.from_xml_path(f'diploma/simple_models/{model_filename}')
#print('a', model.jnt_actfrclimited)
data = mujoco.MjData(model)

print(*list(zip(bodynames(model.names, model.name_bodyadr), model.body_mass)), sep='\n')

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

'''

experiment = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(-180, 180, 60):
        for j in range(-180, 180, 60):
            i = j = 0.001
            data.qpos[0] = (i)/180*np.pi
            data.qpos[1] = (j)/180*np.pi
            data.qvel[0] = 0
            data.qvel[1] = 0
            data.qacc[0] = 0
            data.qacc[1] = 0
            #data.ctrl[act_idxs[0]] = i/180*np.pi
            #data.ctrl[act_idxs[1]] = j/180*np.pi
            #mujoco.mj_step(model, data)
            #mujoco.mj_forward(model, data)
            #mujoco.mj_Euler(model, data)
            #mujoco.mj_inverse(model, data)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            jacp, jacd = np.zeros([3, 2]), np.zeros([3, 2])
            fullm = np.zeros([2, 2])
            mujoco.mj_jac(model, data, jacp, jacd, data.xpos[3], 3) 
            mujoco.mj_fullM(model, fullm, data.qM)
            fulljac = np.vstack([jacp, jacd])
            m_jacPI = fullm @ np.linalg.pinv(fulljac)
            DME = m_jacPI.T @ m_jacPI
            val, vec = np.linalg.eig(DME[:3, :3])
            val[val == 0] = 1e27
            val = np.sqrt(val)
            val = 1 / val
            val /= 10
            print(val, )
            print(vec)
            experiment.append([data.xpos[3][[0, 2]], np.arctan2(vec[:, 0][0], vec[:, 0][2]), val[0], val[1], vec[[0, 2], 0], vec[[0, 2], 1]])
            
            viewer.sync()
            #time.sleep(0.1)
        

plt.figure()
ax = plt.gca()
ax.set_aspect('equal')

for pos, ang, w, h, ax1, ax2 in experiment:
    #ax.add_patch(matplotlib.patches.Ellipse(xy=pos, width=w, height=h, edgecolor='gray', fc='None', lw=1, angle=ang))
    a, b = pos - ax1 * w, pos + ax1 * w
    #print(ax1, w, ax2, h)
    ax.plot([a[0], b[0]], [a[1], b[1]], color='gray')
    a, b = pos - ax2 * h, pos + ax2 * h
    ax.plot([a[0], b[0]], [a[1], b[1]], color='gray')

ax.scatter([i[0][0] for i in experiment], [i[0][1] for i in experiment])
print(set([i[1] for i in experiment]))

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_xlabel('x')
ax.set_ylabel('z')

plt.show()  #'''
