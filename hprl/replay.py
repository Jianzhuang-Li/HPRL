import numpy as np
import os
import gzip
import pickle
from spe.general_utils import AttrDict

class RelayBuffer:
    """Stores arbitrary rollout outputs that provided by AttriDicts.
    """
    def __init__(self, capacity:int=1e6):
        self.capacity = capacity
        self._replay_buffer = None
        self._idx = None
        self._size = None

    def append(self, experence_batch):
        """Append the vals in AttriDict experience_batch to the exist replay buffer.
        """
        # init the replay buffer if it is None.
        if self._replay_buffer is None:
            self._init_buffer(experence_batch)

        # complete the index range
        n_samples = self._get_sample_num(experence_batch)
        idxs = np.asarray(np.arange(self._idx, self._idx + n_samples) % self.capacity, dtype=int)

        # add batch
        for key in self._replay_buffer:
            self._replay_buffer[key][idxs] = np.stack(experence_batch[key])
        
        # advance pointer
        self._idx = int((self._idx + n_samples) % self.capacity)
        self._size = int(min(self._size + n_samples, self.capacity))

    def _init_buffer(self, example_batch):
        """Initializes the replaybuffer fields given an example experience batch.
        """
        self._replay_buffer = AttrDict()
        for key in example_batch:
            example_element = example_batch[key][0]
            # create a empty space
            self._replay_buffer[key] = np.empty([int(self.capacity)]+list(example_element.shape), \
                                                dtype=example_element.dtype)
        self._size = 0
        self._idx = 0

    def reset(self):
        """Deletes all entries from replay buffer and reinitializes.
        """
        del self._replay_buffer
        self._replay_buffer, self._idx, self._size = None, None, None

    def get(self):
        """Return complete replay buffer.
        """
        return self._replay_buffer
    
    def sample(self, n_samples, filter=None):
        """Sample n_samples from the rollout_storage.
        Potentially can filter with fields to return.
        """
        raise NotImplementedError

    def save(self, save_dir):
        """ Store compressed replay buffer to file. 
        """
        os.mkdirs(save_dir, exit=True)
        with gzip.open(os.path.join(save_dir, "replay_buffer.zip"), 'wb') as f:
            pickle.dump(self._replay_buffer, f)
        np.save(os.path.join(save_dir, "idx_size.npy"), np.array([self._idx, self._size]))

    def load(self, save_dir):
        """Load replay buffer from compressed disk file.
        """
        assert self._replay_buffer is None
        with gzip.open(os.path.join(save_dir, "relay_buffer.zip"), 'rb') as f:
            self._replay_buffer = pickle.load(f)
        idx_size = np.load(os.path.join(save_dir, "idx_size.npy"))
        self._idx, self.size = int(idx_size[0]), int(idx_size[1])
    
    @staticmethod
    def _get_sample_num(experience_batch):
        for key in experience_batch:
            return len(experience_batch[key])
    
    @property
    def size(self):
        return self._size
    
    @property
    def idx(self):
        return self._idx

class UniformRelayBuffer(RelayBuffer):

    def __init__(self, capacity: int = 1000000):
        super().__init__(capacity)
    
    def sample(self, n_samples, filter=None):
        assert n_samples <= self.size
        assert isinstance(self.size, int)
        idxs = np.random.choice(np.arange(self.size), size=n_samples)

        sampled_transitions = AttrDict()
        for key in self._replay_buffer:
            if filter is None or key in filter:
                sampled_transitions[key] = self._replay_buffer[key][idxs]
        return sampled_transitions

if __name__ == "__main__":
    # buffer test
    states = np.arange(10)
    actions = np.arange(10)
    experience_batch = AttrDict()
    experience_batch["states"] = states
    experience_batch["actions"] = actions

    buffer = UniformRelayBuffer()
    buffer.append(experience_batch)
    print(buffer.idx)
    print(buffer.size)
    print(buffer.sample(5))