import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from common import *
from jacobians import *
from static_logger import StaticLogger
np.set_printoptions(precision=4, suppress=True)


model = mujoco.MjModel.from_xml_path(f'v1_5bar/5bar.xml')
end_idx = get_endeffector_idx(model)
actjnt_idxs, act_idxs = find_jnt_and_act(model)
jjidxs = J_5bar_get_idxs(model)
end_axis = np.array([0, -1, 0])
data = mujoco.MjData(model)


path = np.array([
    [ 0.9995,  0.0308,  0.,     -0.0006],
    [ 0.9408, -0.0374,  0.0134,  0.3367],
    [ 0.9472, -0.3062, -0.0292, -0.0904],
    [ 0.9438,  0.0076,  0.0027, -0.3304],
    [ 0.9524,  0.2848,  0.0312, -0.1044],
    [ 0.9995,  0.0308,  0.,     -0.0006],
])
path_idx = 0
pos = np.array([0, 0], dtype='float64')


logger = StaticLogger(model, 'test_path_5bar', [0, 1], [0, 1], end_idx, end_axis)

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_check = time.time()
    start = time.time()
    while viewer.is_running() and path_idx < len(path):
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()
        
        J = J_5bar(data, jjidxs, end_idx, end_axis)
        
        qvel = None
        while qvel is None:
            qvel = np.zeros([3, 1])
            mujoco.mju_subQuat(qvel, path[path_idx], data.xquat[end_idx])
            qvel_l = (qvel**2).sum() ** 0.5 
            if qvel_l < 0.05:
                if path_idx < len(path)-1:
                    path_idx += 1
                    qvel = None
        #qvel /= qvel_l

        qvel = (np.linalg.pinv(J) @ (qvel)).reshape([2])
        pos += qvel * model.opt.timestep * 2
        print(path_idx, path[path_idx], data.xquat[end_idx], qvel_l)

        data.ctrl[0] = pos[0]
        data.ctrl[1] = pos[1]

        logger.add_data(data)

        '''time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)'''
            
logger.save_data()

# траектории вокруг осей
# datasert daily human routine mocap and else
# jacobians
# 