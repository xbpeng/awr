import numpy as np
import copy
import util.logger as logger
import util.rl_path as rl_path

INVALID_IDX = -1

class ReplayBuffer(object):

    TERMINATE_KEY = "terminate"
    PATH_START_KEY = "path_start"
    PATH_END_KEY = "path_end"

    def __init__(self, buffer_size):
        assert buffer_size > 0

        self.buffer_size = buffer_size
        self.total_count = 0
        self.buffer_head = 0
        self.buffer_tail = INVALID_IDX
        self.num_paths = 0
        self.buffers = None

        self.clear()
        return

    def sample(self, n, filter_end=True):
        curr_size = self.get_current_size()
        assert curr_size > 0

        if (filter_end):
            idx = np.empty(n, dtype=int)
            # makes sure that the end states are not sampled
            for i in range(n):
                while True:
                    curr_idx = np.random.randint(0, curr_size, size=1)[0]
                    curr_idx += self.buffer_tail
                    curr_idx = np.mod(curr_idx, self.buffer_size)

                    if not self.is_path_end(curr_idx):
                        break
                idx[i] = curr_idx
        else:
            idx = np.random.randint(0, curr_size, size=n)
            idx += self.buffer_tail
            idx = np.mod(idx, self.buffer_size)

        return idx

    def get(self, key, idx):
        return self.buffers[key][idx]

    def get_all(self, key):
        return self.buffers[key]

    def get_unrolled_indices(self):
        indices = None
        if self.buffer_tail == INVALID_IDX:
            indices = []
        elif self.buffer_tail < self.buffer_head:
            indices = list(range(self.buffer_tail, self.buffer_head))
        else:
            indices = list(range(self.buffer_tail, self.buffer_size))
            indices += list(range(0, self.buffer_head))
        return indices
    
    def get_path_start(self, idx):
        return self.buffers[self.PATH_START_KEY][idx]

    def get_path_end(self, idx):
        return self.buffers[self.PATH_END_KEY][idx]

    def get_subpath_indices(self, idx):
        assert(isinstance(idx, int))

        start_idx = idx
        end_idx = self.get_path_end(idx)

        if (start_idx <= end_idx):
            path_indices = list(range(start_idx, end_idx + 1))
        else:
            path_indices = list(range(start_idx, self.buffer_size))
            path_indices += list(range(0, end_idx + 1))

        return path_indices

    def get_pathlen(self, idx):
        is_array = isinstance(idx, np.ndarray) or isinstance(idx, list)
        if not is_array:
            idx = [idx]

        n = len(idx)
        start_idx = self.get_path_start(idx)
        end_idx = self.get_path_end(idx)
        pathlen = np.empty(n, dtype=int)

        for i in range(n):
            curr_start = start_idx[i]
            curr_end = end_idx[i]
            if curr_start < curr_end:
                curr_len = curr_end - curr_start
            else:
                curr_len = self.buffer_size - curr_start + curr_end
            pathlen[i] = curr_len

        if not is_array:
            pathlen = pathlen[0]

        return pathlen

    def get_subpath_indices(self, start_idx):
        end_idx = self.get_path_end(start_idx)

        if start_idx <= end_idx:
            path_indices = list(range(start_idx, end_idx + 1))
        else:
            path_indices = list(range(start_idx, self.buffer_size))
            path_indices += list(range(0, end_idx + 1))

        return path_indices

    def is_valid_path(self, idx):
        start_idx = self.get_path_start(idx)
        valid = start_idx != INVALID_IDX
        return valid

    def store(self, path):
        start_idx = INVALID_IDX
        n = path.pathlength()
        
        if (n > 0):
            assert path.is_valid()

            if path.check_vals():
                if self.buffers is None:
                    self._init_buffers(path)

                idx = self._request_idx(n + 1)
                self._store_path(path, idx)

                self.num_paths += 1
                self.total_count += n + 1
                start_idx = idx[0]
            else:
                logger.Logger.print("Invalid path data value detected")

        return start_idx

    def clear(self):
        self.buffer_head = 0
        self.buffer_tail = INVALID_IDX
        self.num_paths = 0
        return
    
    def get_prev_idx(self, idx):
        prev_idx = idx - 1
        prev_idx[prev_idx < 0] += self.buffer_size
        is_start = self.is_path_start(idx)
        prev_idx[is_start] = idx[is_start]
        return prev_idx

    def get_next_idx(self, idx):
        next_idx = np.mod(idx + 1, self.buffer_size)
        is_end = self.is_path_end(idx)
        next_idx[is_end] = idx[is_end]
        return next_idx

    def is_terminal_state(self, idx):
        terminate_flags = self.buffers[self.TERMINATE_KEY][idx]
        terminate = terminate_flags != rl_path.Terminate.Null.value
        is_end = self.is_path_end(idx)
        terminal_state = np.logical_and(terminate, is_end)
        return terminal_state

    def check_terminal_flag(self, idx, flag):
        terminate_flags = self.buffers[self.TERMINATE_KEY][idx]
        terminate = (terminate_flags == flag.value)
        return terminate

    def is_path_start(self, idx):
        is_end = self.buffers[self.PATH_START_KEY][idx] == idx
        return is_end

    def is_path_end(self, idx):
        is_end = self.buffers[self.PATH_END_KEY][idx] == idx
        return is_end

    def get_current_size(self):
        if self.buffer_tail == INVALID_IDX:
            return 0
        elif self.buffer_tail < self.buffer_head:
            return self.buffer_head - self.buffer_tail
        else:
            return self.buffer_size - self.buffer_tail + self.buffer_head

    def get_valid_idx(self):
        valid_idx = np.argwhere(self.buffers[self.PATH_START_KEY] != INVALID_IDX)
        is_end = self.is_path_end(valid_idx)
        valid_idx = valid_idx[np.logical_not(is_end)]
        return valid_idx

    def _init_buffers(self, path):
        self.buffers = dict()
        self.buffers[self.PATH_START_KEY] = INVALID_IDX * np.ones(self.buffer_size, dtype=int);
        self.buffers[self.PATH_END_KEY] = INVALID_IDX * np.ones(self.buffer_size, dtype=int);
        self.buffers[self.TERMINATE_KEY] = np.zeros(shape=[self.buffer_size], dtype=int)

        for key, val in vars(path).items():
            if type(val) is list:
                val_type = type(val[0])
                is_array = val_type == np.ndarray
                if is_array:
                    shape = [self.buffer_size, val[0].shape[0]]
                    dtype = val[0].dtype
                else:
                    shape = [self.buffer_size]
                    dtype = val_type
                    
                self.buffers[key] = np.zeros(shape, dtype=dtype)
        return

    def _request_idx(self, n):
        assert n + 1 < self.buffer_size # bad things can happen if path is too long

        remainder = n
        idx = []

        start_idx = self.buffer_head
        while remainder > 0:
            end_idx = np.minimum(start_idx + remainder, self.buffer_size)
            remainder -= (end_idx - start_idx)

            free_idx = list(range(start_idx, end_idx))
            self._free_idx(free_idx)
            idx += free_idx
            start_idx = 0

        self.buffer_head = (self.buffer_head + n) % self.buffer_size
        return idx

    def _free_idx(self, idx):
        assert(idx[0] <= idx[-1])
        n = len(idx)
        if self.buffer_tail != INVALID_IDX:
            update_tail = idx[0] <= idx[-1] and idx[0] <= self.buffer_tail and idx[-1] >= self.buffer_tail
            update_tail |= idx[0] > idx[-1] and (idx[0] <= self.buffer_tail or idx[-1] >= self.buffer_tail)
            
            if update_tail:
                i = 0
                while i < n:
                    curr_idx = idx[i]
                    if self.is_valid_path(curr_idx):
                        start_idx = self.get_path_start(curr_idx)
                        end_idx = self.get_path_end(curr_idx)
                        pathlen = self.get_pathlen(curr_idx)

                        if start_idx < end_idx:
                            self.buffers[self.PATH_START_KEY][start_idx:end_idx + 1] = INVALID_IDX
                        else:
                            self.buffers[self.PATH_START_KEY][start_idx:self.buffer_size] = INVALID_IDX
                            self.buffers[self.PATH_START_KEY][0:end_idx + 1] = INVALID_IDX
                        
                        self.num_paths -= 1
                        i += pathlen + 1
                        self.buffer_tail = (end_idx + 1) % self.buffer_size;
                    else:
                        i += 1
        else:
            self.buffer_tail = idx[0]
        return

    def _store_path(self, path, idx):
        n = path.pathlength()
        for key, data in self.buffers.items():
            if key != self.PATH_START_KEY and key != self.PATH_END_KEY and key != self.TERMINATE_KEY:
                val = getattr(path, key)
                val_len = len(val)
                assert val_len == n or val_len == n + 1
                data[idx[:val_len]] = val

        self.buffers[self.TERMINATE_KEY][idx] = path.terminate.value
        self.buffers[self.PATH_START_KEY][idx] = idx[0]
        self.buffers[self.PATH_END_KEY][idx] = idx[-1]
        return