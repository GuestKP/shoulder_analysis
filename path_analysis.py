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


model = mujoco.MjModel.from_xml_path(f'v1_Gimbal/Gimbal_vel.xml')
data = mujoco.MjData(model)
path = np.array([
    [-0.0226, -0.0846, -0.0505],
    [ 0.0668, -0.0742,  0.0157],
    [-0.0321, -0.0912,  0.0293],
    [-0.0553, -0.0836, -0.0126],
    [-0.0324, -0.0926, -0.024 ],
    [-0.0226, -0.0846, -0.0505],
])
path_idx = 0


end_idx = get_endeffector_idx(model)
try:
    actjnt_idxs, act_idxs = find_jnt_and_act(model)
    jjidxs = J_5bar_get_idxs(model)
    end_axis = np.array([-1, 0, 0])
except:
    pass

logger = StaticLogger(model, 'test_path', [0, 1], [0, 1], end_idx)

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_check = time.time()
    start = time.time()
    while viewer.is_running() and path_idx < len(path):
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()
        
        J = J_gimbal(model, data)[:3, :3]
        
        err = None
        while err is None:
            err = path[path_idx] - data.xpos[end_idx]
            err_l = (err**2).sum() ** 0.5 
            if err_l < 0.01:
                path_idx += 1
                if path_idx < len(path)-1:
                    err = None

        pos = np.linalg.pinv(J) @ (err / err_l * 0.1)

        data.ctrl[0] = pos[0]
        data.ctrl[1] = pos[1]

        logger.add_data(data)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
logger.save_data()