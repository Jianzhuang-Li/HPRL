import pandas as pd
import numpy as np
import random
from typing import Dict, List
import torch
from sinergym.utils.logger import Logger
import os
#from torch.utils.data import Dataset, DataLoader
from spirl.components.data_loader import Dataset
from spirl.utils.general_utils import AttrDict



# 自定义数据集类
class EplusDatasetForSkillPrior(Dataset):

    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self,
                 phase:str, # train or val
                 file_path:str,
                 subseq_len:int,    # the length of skill
                 action_names:List[str],    # action names in csv data
                 obs_names:List[str],   # state names in csv data
                 normalized:bool=True,  # use normalized data or raw data
                 last_run_num:int = -1,    # get the last num of sub run data, set -1 to get all
                 set_index_range:bool = False, # if it is true, we get data from index start to end
                 start_run_index:int = 0,
                 end_run_index:int = -1,
                 dataset_size = -1, 
                 prefix:str='Eplus-env-sub',
                 suffix:str='run'):
        self.phase = phase
        self.file_path:str = file_path
        self.subseq_len:int = subseq_len
        self.last_run_num:int = last_run_num
        self.set_index_range:bool = set_index_range
        self.start_run_index = start_run_index
        self.end_run_index = end_run_index
        self.prefix:str = prefix
        self.suffix:str = suffix
        self.normalized:bool = normalized
        self.action_name:list[str] = action_names
        self.obs_name:list[str] = obs_names
        self.dataset_size = dataset_size

        self.obs_dim = len(self.obs_name)
        self.act_dim = len(self.action_name)

        self.trajectories_dir:Dict[int, str] = {}
        self.trajectories_csv = {}
        self.seqs:list[AttrDict] = []
        self.n_seqs = 0

        self.logger = Logger().getLogger("EplusDataSet", "INFO")

        # const val
        self.action_low = np.array([12.0, 23.25], dtype=np.float32)
        self.action_high = np.array([23.25, 30.0], dtype=np.float32)

        self._get_sub_run_map(self.file_path)
        self._get_sub_run_data_csv()
        self._get_data_to_list()
        self._filter_indices()
        self._spilt_data()

    
    def _get_sub_run_map(self, path:str)->None:
        """
        get the subrun data dir.
        """
        assert os.path.exists(path)
        dir_list =os.listdir(path=path)
        self.trajectories_dir.clear()
        for sub_dir in dir_list:
            if os.path.isdir( os.path.join(path, sub_dir)):
                sub_dir_splits = sub_dir.split('_')
                if len(sub_dir_splits) != 2:
                    continue
                if sub_dir_splits[0] != self.prefix:
                    continue
                run_mark = sub_dir_splits[1]
                if len(run_mark) < 4:
                    continue
                suffix = run_mark[0:3]
                if suffix != self.suffix:
                    continue
                try:
                    run_num = int(run_mark[3:])
                    self.trajectories_dir[run_num] = sub_dir
                except ValueError as e:
                    self.logger.info(f"invalid data dir: {sub_dir}")
        if len(self.trajectories_dir.items())==0:
            self.logger.warning(f"no sub_run data find in {self.file_path}")

    def _get_sub_run_data_csv(self):
        assert len(self.trajectories_dir.items()) > 0
        self.trajectories_csv.clear()
        for index, sub_data_dir in self.trajectories_dir.items():
            if self.set_index_range and (index<self.start_run_index or index>=self.end_run_index):
                continue
            else:
                max_index = max(self.trajectories_dir.keys())
                if self.last_run_num != -1 and index < (max_index-self.last_run_num+1):
                    continue
            sub_data_path = os.path.join(self.file_path, sub_data_dir)
            assert os.path.exists(sub_data_path)
            if self.normalized:
                path = os.path.join(sub_data_path, 'monitor_normalized.csv')
            else:
                path = os.path.join(sub_data_path, 'monitor.csv')
            assert os.path.exists(path)
            csv_data = pd.read_csv(path, encoding='utf-8')
            columns_name = csv_data.columns
            for var_name in self.action_name + self.obs_name:
                assert var_name in columns_name
            self.trajectories_csv[index] = csv_data

    def _get_data_to_list(self):
        for index, csv_data in self.trajectories_csv.items():
            self.seqs.append(AttrDict(
                    states = csv_data[self.obs_name].iloc[:].values.astype(np.float32),
                    actions = csv_data[self.action_name].iloc[:].values.astype(np.float32),
                    rewards = csv_data["reward"].iloc[:].values.astype(np.float32),
                    terminals = csv_data["terminated"].iloc[:].values.astype(bool) 
            ))
        # release csv data
        self.trajectories_csv.clear()
        self.n_seqs = len(self.seqs)

    def _pad_n_steps(self, steps):
        assert len(self.seqs) > 0
        for seq in self.seqs:
            seq.states = np.concatenate((np.zeros((steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
            seq.actions = np.concatenate((np.zeros((steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))
            seq.rewards = np.concatenate((np.zeros((steps, seq.rewards.shape[1]), dtype=seq.rewards.dtype), seq.rewards))
            seq.terminals = np.concatenate((np.zeros((steps, seq.terminals.shape[1]), dtype=seq.terminals.dtype), seq.terminals))
        self.n_seqs = len(self.seqs0)

    def _filter_indices(self):
        assert len(self.seqs) > 0
        random.shuffle(self.seqs) # random.shuffle()用于将一个列表中的元素打乱顺序
        self.n_seqs = len(self.seqs)

    def _spilt_data(self):
        self.start = 0
        self.end = self.n_seqs
        """
        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs
        """
        
    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        all_traj_len = 0
        for traj in self.seqs:
            all_traj_len += len(traj.states)
        return int(all_traj_len/self.subseq_len)
        # return int(self.SPLIT[self.phase]*all_traj_len/self.subseq_len)
    
    def __getitem__(self, idx):
        # sample start index in data range
        seq = self._sample_seq()
        start_index = np.random.randint(1, seq.states.shape[0]-self.subseq_len-1)
        states_ = seq.states[start_index:start_index+self.subseq_len]
        actions_ = seq.actions[start_index:start_index+self.subseq_len-1]
        scaled_actions_ = 2 * (actions_ - self.action_low) / (self.action_high - self.action_low) - 1
        output = AttrDict(
            states= states_,
            actions= scaled_actions_
        )
        return output


    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

class EplusSpirlSplitDataset(EplusDatasetForSkillPrior):

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.spec = data_conf.dataset_spec
        self.device = data_conf.device
        self.shuffle = shuffle
        self.n_worker = 4
        EplusDatasetForSkillPrior.__init__(self,
                                            phase=phase,
                                            file_path="/home/jianzhuang/Documents/projects/RLBuilding/data/Eplus-env-SB3-PPO-Eplus-5zone-hot-0728",
                                            subseq_len=self.spec.subseq_len,
                                            last_run_num=5,
                                            action_names=["Heating_Setpoint_RL", "Cooling_Setpoint_RL"],
                                            obs_names=["hour", "outdoor_temperature","outdoor_humidity","wind_speed","wind_direction","diffuse_solar_radiation","direct_solar_radiation", "htg_setpoint", "clg_setpoint", "air_temperature", "air_humidity", \
                                                 "people_occupant", "co2_emission", "HVAC_electricity_demand_rate", "total_electricity_HVAC"],
                                            normalized=True
                                        )


if __name__ == "__main__":
    dataset = EplusDatasetForSkillPrior(phase="train",
                                        file_path="/home/jianzhuang/Documents/projects/RLBuilding/data/Eplus-env-SB3-PPO-Eplus-5zone-hot-0728",
                                        subseq_len=10,
                                        last_run_num=5,
                                        action_names=["Heating_Setpoint_RL", "Cooling_Setpoint_RL"],
                                        obs_names=["hour", "outdoor_temperature","outdoor_humidity","wind_speed","wind_direction","diffuse_solar_radiation","direct_solar_radiation", "htg_setpoint", "clg_setpoint", "air_temperature", "air_humidity", \
                                                 "people_occupant", "co2_emission", "HVAC_electricity_demand_rate", "total_electricity_HVAC"],
                                        normalized=True
                                        )
    print(len(dataset))
    print(dataset[0])
    # print(type(dataset[0].actions))
    print(dataset.seqs[0].rewards.shape)