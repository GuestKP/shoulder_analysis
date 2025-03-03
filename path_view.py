import numpy as np
import matplotlib.pyplot as plt

save_name = 'test_path'

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

plt.plot(x_ax, data_eef_pos[:, 0])
plt.plot(x_ax, data_eef_pos[:, 1])
plt.plot(x_ax, data_eef_pos[:, 2])
plt.show()