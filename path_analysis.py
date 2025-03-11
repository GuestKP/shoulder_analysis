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


model = mujoco.MjModel.from_xml_path(f'v1_Gimbal/Gimbal.xml')
end_idx = get_endeffector_idx(model)
try:
    actjnt_idxs, act_idxs = find_jnt_and_act(model)
    jjidxs = J_5bar_get_idxs(model)
    end_axis = np.array([-1, 0, 0])
except:
    pass
end_axis = np.array([0, -1, 0])
data = mujoco.MjData(model)
path = np.array([
    [ 0.7062, -0.0353,  0.0353,  0.7062],
    [ 0.4395, -0.0449,  0.022,   0.8968],
    [ 0.7523, -0.0944,  0.1105,  0.6426],
    [ 0.8265, -0.0192,  0.0283,  0.5618],
    [ 0.6853,  0.2079, -0.2145,  0.6641],
    [ 0.7062, -0.0353,  0.0353,  0.7062],
])
path_idx = 0
pos = np.array([0, 0], dtype='float64')

logger = StaticLogger(model, 'test_path_gimbal', [0, 1], [0, 1], end_idx, end_axis)

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_check = time.time()
    start = time.time()
    while viewer.is_running() and path_idx < len(path):
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()
        
        J = J_gimbal(model, data)[3:]
        
        err = None
        while err is None:
            err = np.zeros([3, 1])
            mujoco.mju_subQuat(err, path[path_idx], data.xquat[end_idx])
            err_l = (err**2).sum() ** 0.5 
            if err_l < 0.05:
                if path_idx < len(path)-1:
                    path_idx += 1
                    err = None
        #qvel /= qvel_l

        qvel = (np.linalg.pinv(J) @ (err)).reshape([2])
        pos += qvel * model.opt.timestep * 10
        print(path_idx, path[path_idx], data.xquat[end_idx], err.reshape([-1]), err_l, '\n', J)

        data.ctrl[0] = pos[0]
        data.ctrl[1] = pos[1]

        logger.add_data(data)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
logger.save_data()

# траектории вокруг осей
# datasert daily human routine mocap and else
# jacobians
# 