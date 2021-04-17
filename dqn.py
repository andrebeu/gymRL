import collections
import numpy
import gym
from collections import namedtuple
import torch as tr
import gym
import numpy as np

# container 
Experience = namedtuple('Experience',[
    'tstep','state','action','reward','state_tp1'
])


def unpack_expL(expLoD):
    """ given list of exp (namedtups)
    return dict of np.arrs
    """
    expDoL = Experience(*zip(*expLoD))._asdict()
    return {k:np.array(v) for k,v in expDoL.items()}



class Task():

    def __init__(self,task_name='CartPole-v1'):
        """ wrapper for interacting with 
            openai gym control tasks 
        """
        self.env = env = gym.make(task_name)
        self.env.seed(0)
        # used for default random policy
        self.aspace = self.env.action_space.n

    def play(self,policy_fn=None,max_ep_len=1000):
        """ 
        given policy, return trajectory
            pi(s_t) -> a_t
        returns episode_exp = {st:[],at:[],rt:[],spt:[]}
        """        
        self.env.reset()
        # random policy
        if type(policy_fn)==type(None):
            policy_fn = lambda x: np.random.randint(self.aspace)
        # init loop vars
        done = False
        tstep = 0
        env_out = self.env.step(0) # arbitrary a0
        sp_t,r_t,done,extra = env_out
        # init ep obj for collecting data
        episode = []
        while not done:
            tstep += 1 
            s_t = sp_t
            # sample a_t, observe transition
            a_t = policy_fn(s_t)
            env_out_t = self.env.step(a_t)
            sp_t,r_t,done,extra = env_out_t
            # collect transition 
            episode.append(
                Experience(tstep,s_t,a_t,r_t,sp_t)
            )
            # verify if done
            if tstep==max_ep_len:
                done=True
        return episode



class Buffer():
    """ deque with record and sample method
    """

    def __init__(self,size=10000):
        """ buffer is list of dicts
        """
        self.buff_size = size
        self.reset_buff()
        return None

    def reset_buff(self):
        self.buffer = collections.deque(maxlen=self.buff_size)
        return None

    def record(self,episode,verb=False):
        """ record episode
        append list of experiences to buffer
        """
        self.eplen = len(episode)
        self.buffer.extend(episode)
        return None

    def sample(self,mode='rand',nsamples=64):
        """ sample exp from the buffer
        return 
        """
        # consider all samples in buffer
        if mode == 'rand': 
            exp = self.buffer
        # only consider last episode
        elif mode == 'ep': 
            exp = list(self.buffer)[-self.eplen:]
        # return dict of array 
        exp_samples = np.random.choice(exp,nsamples)
        return exp_samples



class DQN(tr.nn.Module):

    def __init__(self):
        super().__init__()
        self.indim = 4 # 4 obs + 1 action
        self.stsize = 20
        self.outdim = 2 # num actions
        self.build()
        self.lossop = tr.nn.SmoothL1Loss()
        self.optiop = tr.optim.Adam(self.parameters(),lr=0.001)
        self.gamma=0.98

    def build(self):
        self.ff1 = tr.nn.Linear(self.indim,self.stsize)
        self.ff2 = tr.nn.Linear(self.stsize,self.stsize)
        self.ff3 = tr.nn.Linear(self.stsize,self.stsize)
        # self.gru = tr.nn.GRU(self.stsize,self.stsize)
        # self.init_rnn = tr.nn.Parameter(tr.rand(2,1,1,self.stsize),requires_grad=True)
        self.out_layer = tr.nn.Linear(self.stsize,self.outdim)

    def forward(self,obs):
        """ 
        takes batch of observations
            obs : arr[batch,sfeat]
        returns value of each action
            qval : arr[batch,nactions]
        """
        h_t = tr.Tensor(obs)
        # h_t = obs_t
        h_t = self.ff1(h_t)
        h_t = self.ff2(h_t).relu()
        h_t = self.ff3(h_t).relu()
        q_val = self.out_layer(h_t)
        return q_val

    def qlearn_loss(self,exp):
        """ 
        exp is dict of arrs 
            {state,action,reward,state_tp1}
        """
        st = exp['state']
        at = exp['action']
        rt = exp['reward']
        stp1 = exp['state_tp1']
        # forward passes
        q_stp1 = self.forward(stp1)
        q_st = self.forward(st)
        ## form ytarget 
        # max_ap{ q(sp_t,ap_t) }
        q_stp1_max_ap = tr.max(q_stp1,1)[0]
        # discount
        ytarget = tr.Tensor(rt) + self.gamma*q_stp1_max_ap
        # final target = reward
        ytarget[-1] = 0
        # print(ytarget,rt)
        # yhat = qvalue of selected actions
        yhat = q_st_at = np.take_along_axis(q_st,at[:,None],axis=1)
        ## episode loss
        ep_loss = self.lossop(ytarget,yhat.squeeze())
        return ep_loss

    def train(self,exp):
        """ 
        exp is dict of arrs
            {state,action,reward,state_tp1}
        """
        self.optiop.zero_grad()
        loss = self.qlearn_loss(exp)
        loss.backward()
        self.optiop.step()
        return None

    def argmax_policy_fn(self,epsilon):
        """ lambda handle for softmax policy
        """

        if np.random.random() > epsilon:
            return lambda x: np.random.randint(2)
        else:
            return lambda x: self.forward(x).argmax().detach().numpy()
