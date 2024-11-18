from torch.utils.tensorboard import SummaryWriter
from collections import deque
from spe.general_utils import prefix_dict
import logging

class TensorBoardLogger(SummaryWriter):

    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix=""):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.logger = logging.getLogger(__name__)

        self.logger_steps = 0
        self.global_steps = 0
        self.logger_interval = 960
        self.cache_size = 100
        self.reward_caches = None
    
    def add_reward(self, reward, steps):
        if self.reward_caches is None:
            self.reward_caches = deque(maxlen=self.cache_size)
        self.reward_caches.append(reward)
        self.logger_steps += steps
        self.global_steps += steps
        if self.logger_steps >= self.logger_interval:
            avg_reward = sum(self.reward_caches)/len(self.reward_caches)
            self.add_scalar(tag='avg_reward', scalar_value=avg_reward, global_step=self.global_steps)
            self._reset_cache()

    def _reset_cache(self):
        self.reward_caches.clear()
        self.logger_steps = 0


    def log_scalar_dict(self, d, prefix='', step=None):
        if prefix: d = prefix_dict(d, prefix + '_')
        if step is None:
            self.logger.warning("step is None")
        else:
            self.logger.info(f"loger out step {step}")  
        for key in d.keys():
            
            self.add_scalar(tag=key, scalar_value=d[key], global_step=step)

