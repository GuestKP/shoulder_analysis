import mujoco
from common import *
import os

class StaticLogger:
    def __init__(self, model, save_name, act_idxs, actjnt_idxs, eef_idx):
        self.model = model
        self.act_idxs = act_idxs
        self.actjnt_idxs = actjnt_idxs
        self.save_name = save_name
        self.eef_idx = eef_idx
        self.reset_data()
        
    def reset_data(self):
        self.data_act_pos = []
        self.data_act_frc = []
        self.data_eef_pos = []
        self.data_dme_val = []
        self.data_dme_vec = []

    def add_data(self, data, do_dme=False):
        mujoco.mj_inverse(self.model, data)
        self.data_act_pos.append([
            data.qpos[self.actjnt_idxs[0]],
            data.qpos[self.actjnt_idxs[1]]
        ])
        self.data_act_frc.append(
            data.qfrc_inverse[self.actjnt_idxs] /
            self.model.actuator_gear[self.act_idxs, 0]
        )
        self.data_eef_pos.append(np.array(data.xpos[self.eef_idx]))

        if do_dme:
            val, vec = DME(self.model, data, data.xpos[self.eef_idx], True)
            self.data_dme_val.append(val)
            self.data_dme_vec.append(vec)

    def save_data(self):
        os.makedirs(f'.data/{self.save_name}', exist_ok=True)
        for arr, fname in [[self.data_act_pos, 'act_pos'],
                          [self.data_act_frc, 'act_frc'],
                          [self.data_eef_pos, 'eef_pos'],
                          [self.data_dme_val, 'dme_val'],
                          [self.data_dme_vec, 'dme_vec']]:
            np.savetxt(f'.data/{self.save_name}/{fname}.csv', arr, delimiter=',')