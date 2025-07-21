import copy
import numpy as np
import quadprog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim import SGD
from torch.utils.data import DataLoader
from .backbone import resnet18
from .base import *


class Replay(data.Dataset):
    """
    A dataset wrapper used as memory for DER++
    """
    def __init__(self, buffer_size, dim_x, device):
        super(Replay, self).__init__()
        self.dim_x = dim_x
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = {}

    def __len__(self):
        if not self.buffer:
            return 0
        n = 0
        for t in self.buffer:
            n += min(self.buffer[t]['num_seen'], self.buffer_size)
        return n

    def add(self, x, y, h, t):
        """
        Add samples to buffer, maintaining memory limits
        """
        x = x.cpu()
        y = y.cpu()
        h = h.cpu()

        if self.dim_x is None:
            self.dim_x = list(x.shape[1:])
            print(f"[MEMORY INIT] Detected input shape: {self.dim_x} (from batch)")

        dim_h = h.shape[-1]

        if t not in self.buffer:
            self.buffer[t] = {
                "X": torch.zeros([self.buffer_size] + self.dim_x),
                "Y": torch.zeros(self.buffer_size).long(),
                "H": torch.zeros([self.buffer_size, dim_h]),
                "num_seen": 0
            }

        n = x.shape[0]
        for i in range(n):
            self.buffer[t]['num_seen'] += 1

            if self.buffer[t]['num_seen'] <= self.buffer_size:
                idx = self.buffer[t]['num_seen'] - 1
            else:
                rand = np.random.randint(0, self.buffer[t]['num_seen'])
                idx = rand if rand < self.buffer_size else -1
                if idx < 0:
                    continue  # skip overflow

            self.buffer[t]['X'][idx] = x[i]
            self.buffer[t]['Y'][idx] = y[i]
            self.buffer[t]['H'][idx] = h[i]

    def sample(self, n, exclude=[]):
        if len(self) == 0:
            return None, None

        X, Y, H = [], [], []
        for t, v in self.buffer.items():
            if t in exclude:
                continue
            n_avail = min(v['num_seen'], self.buffer_size)
            idx = torch.randperm(n_avail)[:min(n, n_avail)]
            X.append(v['X'][idx])
            Y.append(v['Y'][idx])
            H.append(v['H'][idx])
        return torch.cat(X, 0).to(self.device), torch.cat(Y, 0).to(self.device), torch.cat(H, 0).to(self.device)

    def sample_task(self, n, task_id):
        assert task_id in self.buffer, f"[ERROR] Task {task_id} not found in buffer"
        v = self.buffer[task_id]
        n_avail = min(v['num_seen'], self.buffer_size)
        idx = torch.randperm(n_avail)[:min(n, n_avail)]
        return (
            v['X'][idx].to(self.device),
            v['Y'][idx].to(self.device),
            v['H'][idx].to(self.device)
        )

    def remove(self, t):
        if t in self.buffer:
            X = self.buffer[t]['X']
            Y = self.buffer[t]['Y']
            H = self.buffer[t]['H']
            del self.buffer[t]
            print(f"[MEMORY] Removed task {t} from buffer.")
            return X, Y, H
        else:
            print(f"[MEMORY] Tried to remove task {t} which wasn't in buffer.")
            return None, None, None


class Derpp(Base):
    def __init__(self, config):
        super(Derpp, self).__init__(config)
        self.n_mem = config.memory_size if hasattr(config, 'memory_size') else 2000
        self.dim_x = config.dim_input if hasattr(config, 'dim_input') else None
        self.memory = Replay(self.n_mem, self.dim_x, self.device)
        self.alpha = config.alpha
        self.beta = config.beta

    def der_loss(self, a, b, task_id):
        if self.config.scenario != 'domain':
            cpt = self.cpt
            a_ = a[..., task_id * cpt:(task_id + 1) * cpt]
            b_ = b[..., task_id * cpt:(task_id + 1) * cpt]
            loss = F.mse_loss(a_, b_)
        else:
            loss = F.mse_loss(a, b)
        return loss

    def learn(self, task_id, loader):  # Updated to accept DataLoader
        self.opt = SGD(self.net.parameters(),
                       lr=self.config.lr,
                       momentum=self.config.momentum,
                       weight_decay=self.config.weight_decay)

        self.n_iters = self.config.n_epochs * len(loader)

        for epoch in range(self.config.n_epochs):
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                if self.dim_x is None:
                    self.dim_x = list(x.shape[1:])
                    self.memory.dim_x = self.dim_x
                    print(f"[MODEL INIT] Inferred input dims: {self.dim_x}")

                h = self.forward(x, task_id)
                loss = self.loss_fn(h, y)

                n_prev_tasks = len(self.prev_tasks)
                for t in self.prev_tasks:
                    x_past, y_past, h_past = self.memory.sample_task(self.config.mem_batch_size // n_prev_tasks, t)
                    h_tmp = self.forward(x_past, t)
                    loss += self.alpha * self.der_loss(h_tmp, h_past, t) / n_prev_tasks
                    loss += self.beta * self.loss_fn(h_tmp, y_past) / n_prev_tasks

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.memory.add(x, y, h.detach(), task_id)

        self.prev_tasks.append(task_id)

    def forget(self, task_id):
        self.memory.remove(task_id)
        self.prev_tasks.remove(task_id)

        self.opt = SGD(self.net.parameters(),
                       lr=self.config.lr,
                       momentum=self.config.momentum,
                       weight_decay=self.config.weight_decay)

        n_prev_tasks = len(self.prev_tasks)
        for i in range(self.n_iters):
            self.opt.zero_grad()
            loss = 0.0
            for t in self.prev_tasks:
                x_past, y_past, h_past = self.memory.sample_task(self.config.mem_batch_size // n_prev_tasks, t)
                h_tmp = self.forward(x_past, t)
                loss += self.der_loss(h_tmp, h_past, t) / n_prev_tasks
                loss += self.loss_fn(h_tmp, y_past) / n_prev_tasks
            loss.backward()
            self.opt.step()
