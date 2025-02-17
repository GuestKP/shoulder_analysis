import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from common import normalize

foldername = '.data/'
filename = [
    'Gimbal.xml',
    '5bar.xml'
][1]
[pos_end_x, pos_end_y, pos_end_z, data_i, data_j, data_if, data_jf] = np.load(foldername+filename+'.data.npy')
dme_val = np.load(foldername+filename+'.dme_val.npy')
dme_vec = np.load(foldername+filename+'.dme_vec.npy')

pos_end = np.array([pos_end_x, pos_end_y, pos_end_z]).T
pos_end = pos_end / np.abs(pos_end).max()
line_1 = pos_end + dme_vec[:, 0, :] * dme_val[0]

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(data_i, data_j, data_if, marker='o')
ax.scatter(data_i, data_j, data_jf, marker='^')

ax.set_title(filename)
ax.set_xlabel('pitch')
ax.set_ylabel('roll')
ax.set_zlabel('force')

plt.show()

data_i_n = normalize(np.array(data_i, dtype='float64'))
data_j_n = normalize(np.array(data_j, dtype='float64'))


c = np.vstack((data_i_n, data_j_n, np.zeros(len(data_j)))).T

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(*pos_end.T, marker='.', c=c)

ax.set_xlim3d(left=-1, right=1)
ax.set_ylim3d(bottom=-1, top=1)
ax.set_zlim3d(bottom=-1, top=1)
ax.set_title(filename)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

ax = plt.figure().add_subplot(projection='3d')

for i in range(len(pos_end)):
    for axis in 0, 1, 2:
        if dme_val[i, axis] < 1:
            posa, posb = pos_end[i] - dme_vec[i, axis, :] * dme_val[i, axis] * 10, pos_end[i] + dme_vec[i, axis, :] * dme_val[i, axis] * 10
            #print(posa, posb)
            ax.plot(*np.vstack((posa, posb)).T)
        else:
            pass
            #posa, posb = pos_end[i] - dme_vec[i, axis, :] * 0.1, pos_end[i] + dme_vec[i, axis, :] * 0.1
            #print(posa, posb)
            #ax.plot(*np.vstack((posa, posb)).T, color='black')
    #break
ax.set_xlim3d(left=-1, right=1)
ax.set_ylim3d(bottom=-1, top=1)
ax.set_zlim3d(bottom=-1, top=1)
plt.show()