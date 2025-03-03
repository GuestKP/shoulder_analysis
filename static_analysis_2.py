import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from common import *
from os import mkdir
from time import sleep
np.set_printoptions(precision=4, suppress=True)

# only saves data to subfolder. use "view" to visualization

# choose model
model_id = 3
model_filename = [
    'simple_models/gimbal.xml',
    'simple_models/5bar.xml',
    'v1_Gimbal/Gimbal.xml',
    'v1_5bar/5bar.xml',
][model_id]

model = mujoco.MjModel.from_xml_path(model_filename)

# find joints and actuators which I should use
actjnt_idxs, act_idxs = find_jnt_and_act(model)
# check if model has equalities
constrainted = is_constrained(model)

data = mujoco.MjData(model)
'''print(names(model, 'body'), sep='\n')
print(names(model, 'joint'), sep='\n')
print(names(model, 'equality'), sep='\n')
print(names(model, 'actuator'), sep='\n')'''

data.ctrl[act_idxs[0]] = 0/180*np.pi * model.actuator_gear[act_idxs[0], 0]
data.ctrl[act_idxs[1]] = 0/180*np.pi * model.actuator_gear[act_idxs[0], 0]

data.qpos[actjnt_idxs[0]] = 50/180*np.pi * model.actuator_gear[act_idxs[0], 0]
data.qpos[actjnt_idxs[1]] = 50/180*np.pi * model.actuator_gear[act_idxs[0], 0]

data_i = []
data_j = []
data_if = []
data_jf = []
 
pos_end_x = []
pos_end_y = []
pos_end_z = []

dme_val = []
dme_vec = []

range1 = list(range(-180, 181, 10))
range2 = list(range(-180, 181, 10))
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range1:
        print(i)
        for j in (range2 if ((i // 10) % 2) else range2[::-1]):
            '''if model_id == 1:
                if abs(i-j) > 140:
                    continue'''

            data.ctrl[act_idxs[0]] = i/180*np.pi
            data.ctrl[act_idxs[1]] = j/180*np.pi

            to_save = False
            if constrainted:
                # try to sim until converged (unstable)
                if converge_constraint(model, data, viewer):
                    to_save = True
            else:
                # only set positions
                data.qacc *= 0
                data.qpos[actjnt_idxs[0]] = i/180*np.pi
                data.qpos[actjnt_idxs[1]] = j/180*np.pi
                to_save = True
                
            # if converged / data is ok
            if to_save:
                mujoco.mj_inverse(model, data)
                qfrc_actuator = data.qfrc_inverse[actjnt_idxs] / model.actuator_gear[act_idxs, 0]
                pos_end_x.append(data.xpos[-1][0])
                pos_end_y.append(data.xpos[-1][1])
                pos_end_z.append(data.xpos[-1][2])
                data_i.append(i)
                data_j.append(j)
                data_if.append(qfrc_actuator[0])
                data_jf.append(qfrc_actuator[1])

                val, vec = DME(model, data, data.xpos[-1], True)
                dme_val.append(val)
                dme_vec.append(vec)
                #viewer.sync()
                #print(data.xpos[-1])
                #sleep(0.1)


try:
    mkdir('.data/')
except FileExistsError:
    pass
np.save('.data/'+model_filename.split('/')[-1]+'.data.npy', np.array([pos_end_x, pos_end_y, pos_end_z, data_i, data_j, data_if, data_jf]))
np.save('.data/'+model_filename.split('/')[-1]+'.dme_val.npy', np.array(dme_val))
np.save('.data/'+model_filename.split('/')[-1]+'.dme_vec.npy', np.array(dme_vec))


