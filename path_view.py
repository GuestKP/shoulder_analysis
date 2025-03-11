import numpy as np
import matplotlib.pyplot as plt

def sphere(ax, size, freq=20):
    u = np.linspace(0, 2 * np.pi, freq)
    v = np.linspace(0, np.pi, freq)
    x = size * np.outer(np.cos(u), np.sin(v))
    y = size * np.outer(np.sin(u), np.sin(v))
    z = size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='gray')

save_name = 'test_path_5bar'

data_act_pos = np.loadtxt(f'.data/{save_name}/act_pos.csv', delimiter=',')
data_act_frc = np.loadtxt(f'.data/{save_name}/act_frc.csv', delimiter=',')
data_eef_pos = np.loadtxt(f'.data/{save_name}/eef_pos.csv', delimiter=',')

x_ax = range(data_act_pos.shape[0])

plt.plot(x_ax, data_act_pos[:, 0])
plt.plot(x_ax, data_act_pos[:, 1])
plt.show()

plt.plot(x_ax, data_act_frc[:, 0])
plt.plot(x_ax, data_act_frc[:, 1])
plt.show()


ax = plt.figure().add_subplot(projection='3d')
sphere(ax, 0.95)

ax.set_aspect('equal')

ax.scatter(*data_eef_pos[::20].T, marker='.')
ax.set_xlim3d(left=-1, right=1)
ax.set_ylim3d(bottom=-1, top=1)
ax.set_zlim3d(bottom=-1, top=1)

'''plt.plot(x_ax, data_eef_pos[:, 0])
plt.plot(x_ax, data_eef_pos[:, 1])
plt.plot(x_ax, data_eef_pos[:, 2])'''
plt.show()