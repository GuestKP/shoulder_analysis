import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from common import *
from jacobians import *
np.set_printoptions(precision=4, suppress=True)

# opens model and periodically shows info
# use it for W.I.P.

model = mujoco.MjModel.from_xml_path(f'v1_5bar/5bar.xml')
model = mujoco.MjModel.from_xml_path(f'simple_models/5bar.xml')
data = mujoco.MjData(model)

actjnt_idxs, act_idxs = find_jnt_and_act(model)
jjidxs = J_5bar_get_idxs(model)
end_idx = get_endeffector_idx(model)
end_axis = np.array([-1, 0, 0])

pos = np.array([0, 0], dtype='float')
success = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_check = time.time()
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()
        
        J = J_5bar(data, jjidxs, end_idx, np.array([1, 0, 0]))
        pos += np.linalg.pinv(J) @ np.array([0, 0, 1]) * model.opt.timestep
        print(pos)
        data.ctrl[act_idxs[0]] = pos[0]
        data.ctrl[act_idxs[1]] = pos[1]

        if time.time() - last_check >= 1:
            last_check = time.time()
            
            '''print(J)
            print(J @ np.array([1, 1]))
            print(np.linalg.pinv(J) @ np.array([0, 1, 0]))'''
            if time.time() - start > 5:
                pos += J.T @ np.array([0, -0.01, 0])
                print(pos)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            