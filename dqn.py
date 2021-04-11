import torch as tr
import numpy as np

class DQN(tr.nn.Module):

    def __init__(self):
        super().__init__()
        self.indim = 4 # state
        self.stsize = 10
        self.outdim = 2 # num actions
        self.build()

    def build(self):
        self.ff1 = tr.nn.Linear(self.indim,self.stsize)
        self.ff2 = tr.nn.Linear(self.stsize,self.stsize)
        self.ff3 = tr.nn.Linear(self.stsize,self.stsize)
        # self.gru = tr.nn.GRU(self.stsize,self.stsize)
        # self.init_rnn = tr.nn.Parameter(tr.rand(2,1,1,self.stsize),requires_grad=True)
        self.out_layer = tr.nn.Linear(self.stsize,self.outdim)

    def forward(self,obs_t):
        h_t = tr.Tensor(obs_t).view(1,-1)
        # h_t = obs_t
        h_t = self.ff1(h_t)
        h_t = self.ff2(h_t).relu()
        h_t = self.ff3(h_t).relu()
        q_val = self.out_layer(h_t)
        return q_val
