import collections
import numpy
import gym
from collections import namedtuple
import torch as tr
import gym
import numpy as np

REWARD = [0,1,-1]

# container 
Experience = namedtuple('Experience',[
    'tstep','state','action','reward','state_tp1'
])


class Task():

    def __init__(self,task_name='CartPole-v1',env_seed=0,max_ep_len=100):
        """ wrapper for interacting with 
            openai gym control tasks 
        """
        self.env = env = gym.make(task_name)
        self.env.seed(env_seed)
        # used for default random policy
        self.aspace = self.env.action_space.n
        self.rand_policy = lambda x: np.random.randint(self.aspace)
        self.max_ep_len = max_ep_len 

    def play_ep(self,policy_fn=None):
        """ 
        given policy, return trajectory
            pi(s_t) -> a_t
        returns episode_exp = {st:[],at:[],rt:[],spt:[]}
        """        
        # config env
        self.env.reset()
        if type(policy_fn)==type(None):
            policy_fn = self.rand_policy
        # init loop vars
        done = False
        tstep = 0
        env_out = self.env.step(0) 
        sp_t,r_t,done,extra = env_out
        episode = []
        # episode loop
        while not done:
            tstep += 1 
            s_t = sp_t
            # sample a_t, observe transition
            a_t = policy_fn(s_t)
            env_out_t = self.env.step(a_t)
            # preprocessing from Mnih d.t. apply
            sp_t,r_t,done,extra = env_out_t
            # collect transition 
            episode.append(
                Experience(tstep,s_t,a_t,r_t,sp_t)
            )
            # verify if done
            if tstep==self.max_ep_len:
                done=True
        return episode



class Buffer():
    """ deque with record and sample method
    """

    def __init__(self,mode,size):
        """ buffer is list of dicts
        """
        self.size = size
        self.mode = mode
        self.reset_buff()
        return None

    def reset_buff(self):
        self.bufferL = collections.deque(maxlen=self.size)
        return None

    def record(self,episode,verb=False):
        """ 
        record episodeL of experiences
        extend in bufferL
        """
        # print('record',len(episode))
        # print(episode)
        self.eplen = len(episode)
        self.bufferL.extend(episode)
        return None

    def sample(self,batch_size):
        """ sample experience {t,s,a,r,sp} 
        from the bufferL of exp
        return sampleL 
        """
        # consider all samples in bufferL
        if self.mode == 'online': 
            exp_set = list(self.bufferL)
        # only consider recent steps
        elif self.mode == 'episodic': 
            exp_set = list(self.bufferL)[-self.eplen:]
        # sample from exp_set; bottlneck
        batch_size = self.eplen # controls
        exp_samples = [exp_set[s] for s in \
            np.random.choice(
                np.arange(len(exp_set)),
                batch_size
            )]
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



def unpack_expL(expLoD):
    """ 
    given list of experience (namedtups)
        expLoD [{t,s,a,r,sp}_t]
    return dict of np.arrs 
        exp {s:[],a:[],r:[],sp:[]}
    """
    expDoL = Experience(*zip(*expLoD))._asdict()
    return {k:np.array(v) for k,v in expDoL.items()}

