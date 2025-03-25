import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from common import *
from jacobians import *
from markers import *
np.set_printoptions(precision=4, suppress=True)

# opens model and periodically shows info
# use it for W.I.P.

#model = mujoco.MjModel.from_xml_path(f'v1_5bar/5bar.xml')
#model = mujoco.MjModel.from_xml_path(f'simple_models/5bar.xml')
model = mujoco.MjModel.from_xml_path(f'v1_Gimbal/Gimbal.xml')
print(*names(model), sep='\n')
data = mujoco.MjData(model)

end_idx = get_endeffector_idx(model)
try:
    actjnt_idxs, act_idxs = find_jnt_and_act(model)
    jjidxs = J_5bar_get_idxs(model)
    end_axis = np.array([-1, 0, 0])
except:
    pass

target = np.array([ 0.9935, -0.1135,  0.,     -0.    ])
target = np.array([ 0.7025, -0.0802,  0.0802,  0.7025])

mujoco.mj_forward(model, data)

qneg, qdif, qvel, qtargz, qcurz = [np.zeros([4]) for _ in range(5)]
eef_zero_neq = np.array(data.xquat[end_idx])
mujoco.mju_negQuat(eef_zero_neq, eef_zero_neq)
print(eef_zero_neq)


pos = np.array([0, 0], dtype='float')
success = 0

marker_zero = np.array([0.2, 0, 0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_check = time.time()
    start = time.time()

    marker_drawer = MarkerDrawer(viewer)
    axes_markers = [
        AxesMarker(np.array([0.2, 0, 0]), size_add=0.005),
        AxesMarker(np.array([0.2, 0, 0]), size_add=0.005, colored=False),
        AxesMarker(np.array([0.4, 0, 0]), size_add=0.005),
        AxesMarker(np.array([0.4, 0, 0]), size_add=0.005, colored=False),
        AxesMarker(np.array([0.3, 0.2, 0]), size_add=0.005, colored=False),
    ]
    for ax in axes_markers:
        marker_drawer.markers.extend(ax.markers)

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)

        qcur = data.xquat[end_idx].reshape([4, 1])
        mujoco.mju_mulQuat(qcurz, qcur, eef_zero_neq)
        mujoco.mju_mulQuat(qtargz, target, eef_zero_neq)
        
        mujoco.mju_negQuat(qneg, qcurz)
        mujoco.mju_mulQuat(qdif, qneg, qtargz)

        axes_markers[0].update(data.xquat[end_idx])
        axes_markers[1].update(target)

        axes_markers[2].update(qcurz)
        axes_markers[3].update(qtargz)
        marker_drawer.draw_markers()

        viewer.sync()

        if time.time() - last_check >= 3:
            last_check = time.time()
            J = J_gimbal(model, data)[3:]
            
            print(data.xquat[end_idx])
            # print(J)

            print(data.xquat[end_idx], Rotation.from_quat(data.xquat[end_idx], scalar_first=True).as_euler('xyz'))
            print(qdif, Rotation.from_quat(qdif, scalar_first=True).as_euler('xyz'))

            '''if time.time() - start > 5:
                
                print(pos)'''

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            