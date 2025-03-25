import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import matplotlib.pyplot as plt
import matplotlib
from common import *
from jacobians import *
from static_logger import StaticLogger
from markers import *
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
    [ 0.9987, -0.05,   -0.,      0.    ],
    [ 0.9175,  0.0263,  0.0114,  0.3968],
    [ 0.9588, -0.2548,  0.0322, -0.1211],
    [ 0.9355, -0.0025,  0.0009, -0.3533],
    [ 0.9542,  0.2987,  0.0047,  0.015 ],
    [ 0.9987, -0.05,   -0.,      0.    ],
])
target_idx = len(path)
path_times = [0, 2, 4, 6, 8, 10]

interpolation = Slerp(path_times, Rotation.from_quat(path, scalar_first=True))

path_idx = 0
pos = np.array([0, 0], dtype='float64')

logger = StaticLogger(model, 'test_path_gimbal', [0, 1], [0, 1], end_idx, end_axis)


with mujoco.viewer.launch_passive(model, data) as viewer:
    drawer = MarkerDrawer(viewer)
    path_marks = drawer.mark_path(path, end_axis * 0.1, size=0.01)
    for i in path_marks:
        i.rgba[-1] = 0.3
    drawer.markers.append(
        MarkerData(
            "",
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array([0.004, 0.004, 0.004]),  # size
            interpolation(0.5).as_matrix() @ end_axis * 0.1,  # pos
            rgba=np.array([0.4, 0.8, 0.4, 0.4])
        )
    )
    prev_time_marked, mark_timestep = 0, 0.1

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        #drawer.color_path(path_marks, path_idx)
        if data.time - prev_time_marked > mark_timestep:
            prev_time_marked += mark_timestep
            drawer.markers.append(
                MarkerData(
                    "",
                    mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([0.002, 0.002, 0.002]),  # size
                    Rotation.from_quat(data.xquat[end_idx], scalar_first=True).as_matrix() @ end_axis * 0.1,  # pos
                    rgba=np.array([0.4, 0.8, 0.4, 0.4])
                )
            )
        
        J = J_gimbal(model, data)[3:]
        
        '''if path_idx < len(path):
            err = None
            while err is None and path_idx < len(path):
                err = np.zeros([3, 1])
                mujoco.mju_subQuat(err, path[path_idx], data.xquat[end_idx])

                err_l = (err**2).sum() ** 0.5 
                if err_l < 0.05:
                    path_idx += 1
                    err = None

            if err is not None:
                qvel = (np.linalg.pinv(J) @ (err)).reshape([2])
                pos += qvel * model.opt.timestep * 2

                data.ctrl[0] = pos[0]
                data.ctrl[1] = pos[1]

                logger.add_data(data)'''
        
        _time = data.time
        if _time < path_times[-1]:
            drawer.markers[target_idx].pos = interpolation(_time).as_matrix() @ end_axis * 0.1
            err = np.zeros([3, 1])
            mujoco.mju_subQuat(err, interpolation(_time).as_quat(scalar_first=True), data.xquat[end_idx])
            qvel = (np.linalg.pinv(J) @ (err)).reshape([2])
            print(qvel)
            pos += qvel * model.opt.timestep * 5

            data.ctrl[0] = pos[0]
            data.ctrl[1] = pos[1]

            logger.add_data(data)


        drawer.draw_markers()
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
logger.save_data()

# пофиксить контрол - добавить формулы для desired положения
# 