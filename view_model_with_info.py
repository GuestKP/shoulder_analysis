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

#model = mujoco.MjModel.from_xml_path(f'v1_5bar/5bar.xml')
#model = mujoco.MjModel.from_xml_path(f'simple_models/5bar.xml')
#model = mujoco.MjModel.from_xml_path(f'v1_Gimbal/Gimbal.xml')
model = mujoco.MjModel.from_xml_path(f'v1_Gimbal/Gimbal.xml')
data = mujoco.MjData(model)

end_idx = get_endeffector_idx(model)
try:
    actjnt_idxs, act_idxs = find_jnt_and_act(model)
    jjidxs = J_5bar_get_idxs(model)
    end_axis = np.array([-1, 0, 0])
except:
    pass

pos = np.array([0, 0], dtype='float')
success = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_check = time.time()
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()
        
        J = J_gimbal(model, data)[:3, :3]
        target = np.array([0.0722, 0.0012, 0.1497]) * 2
        err = target - data.xpos[end_idx]
        #err /= (err**2).sum() ** 0.125
        pos = np.linalg.pinv(J) @ err

        if time.time() - last_check >= 1:
            last_check = time.time()
            #J = J_5bar(data, jjidxs, end_idx, np.array([1, 0, 0]))
            
            print(J)
            print(data.xpos[3])
            '''if time.time() - start > 5:
                
                print(pos)'''

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            