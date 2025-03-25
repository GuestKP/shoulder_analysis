import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Slerp, Rotation
from common import *
from jacobians import *
from markers import *
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
target_idx = len(path)
path_times = [0, 3, 6, 9, 12, 15]
path_idx = 0
pos = np.array([0, 0], dtype='float64')

interpolation = Slerp(path_times, Rotation.from_quat(path, scalar_first=True))

logger = StaticLogger(model, 'test_path_5bar', [0, 1], [0, 1], end_idx, end_axis)

with mujoco.viewer.launch_passive(model, data) as viewer:
    drawer = MarkerDrawer(viewer)
    path_marks = drawer.mark_path(path, end_axis * 0.1, size=0.01)
    drawer.markers.append(
        MarkerData(
            "",
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array([0.004, 0.004, 0.004]),  # size
            interpolation(0.5).as_matrix() @ end_axis * 0.1,  # pos
            rgba=np.array([0.8, 0.4, 0.4, 0.4])
        )
    )
    drawer.markers.append(
        MarkerData(
            "",
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array([0.004, 0.004, 0.004]),  # size
            interpolation(0.5).as_matrix() @ end_axis * 0.1,  # pos
            rgba=np.array([0.4, 0.8, 0.4, 0.4])
        )
    )
    prev_time_marked, mark_timestep = 0, 0.2
    
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        J = J_5bar(data, jjidxs, end_idx, end_axis)
        
        '''if path_idx < len(path):
            qvel = None
            while qvel is None and path_idx < len(path):
                qvel = np.zeros([3, 1])
                mujoco.mju_subQuat(qvel, path[path_idx], data.xquat[end_idx])
                qvel_l = (qvel**2).sum() ** 0.5 
                if qvel_l < 0.05:
                    path_idx += 1
                    qvel = None

            if qvel is not None:
                qvel = (np.linalg.pinv(J) @ (qvel * 10)).reshape([2])
                pos += qvel * model.opt.timestep * 0.4
                #print(path_idx, path[path_idx], data.xquat[end_idx], qvel_l)'''

        _time = data.time
        if _time < path_times[-1]:
            drawer.markers[target_idx].pos = interpolation(_time).as_matrix() @ end_axis * 0.1
            err = np.zeros([3, 1])
            mujoco.mju_subQuat(err, interpolation(_time).as_quat(scalar_first=True), data.xquat[end_idx])
            qvel = (np.linalg.pinv(J) @ (err)).reshape([2])
            print(qvel)
            pos += qvel * model.opt.timestep * 10 * 2

            data.ctrl[0] = pos[0]
            data.ctrl[1] = pos[1]

            logger.add_data(data)

            
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
                drawer.markers.append(
                    MarkerData(
                        "",
                        mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([0.002, 0.002, 0.002]),  # size
                        interpolation(_time).as_matrix() @ end_axis * 0.1,  # pos
                        rgba=np.array([0.8, 0.4, 0.4, 0.4])
                    )
                )

        drawer.markers[target_idx+1].pos = Rotation.from_quat(data.xquat[end_idx], scalar_first=True).as_matrix() @ end_axis * 0.1
        drawer.draw_markers()
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
logger.save_data()

# траектории вокруг осей
# datasert daily human routine mocap and else
# jacobians
# 