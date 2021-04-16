import numpy
import gym
from collections import namedtuple
import torch as tr
import gym
import numpy as np



class Task():

    def __init__(self,task_name='CartPole-v1'):
        """ wrapper for interacting with 
            openai gym control tasks 
        """
        self.env = env = gym.make(task_name)
        # used for default random policy
        self.aspace = self.env.action_space.n

    def sample(self,policy_fn=None):
        """ 
        given policy, return trajectory
            pi(s_t) -> a_t
        """
        max_ep_len = 1000
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
        episode = Episode()
        while not done:
            tstep += 1 
            s_t = sp_t
            # sample a_t, observe transition
            a_t = policy_fn(s_t)
            env_out_t = self.env.step(a_t)
            # collect transition 
            sp_t,r_t,done,extra = env_out_t
            episode.record_trans(s_t,a_t,r_t,sp_t)
            # verify if done
            if tstep>=max_ep_len:
                done=True
        return episode


class Episode():

    """ helper object 
    for recording/manipulating
    trajectories in episode
    """

    def __init__(self):
        self.trajectory = []
        None

    def record_trans(self,st,at,rt,spt):
        transition = {
            'st':st,'at':at,
            'rt':rt,'spt':spt
        }
        self.trajectory.append(transition)
        return None

    def format_out(self):
        """ 
        reformats trajectory in buffer 
        from list of dicts, to dict of np.arrs
        """
        traj = {
            'st':[],'at':[],
            'rt':[],'spt':[]
        }
        # loop over transitions in trajectory
        for trans in self.trajectory:
            traj['st'].append(trans['st'])
            traj['at'].append(trans['at'])
            traj['rt'].append(trans['rt'])
            traj['spt'].append(trans['spt'])
        # list to nparr
        traj = {k:np.array(v) for k,v in traj.items()}
        return traj


class Buffer():
    """ deque with sample method
    """

    def __init__(self,nsamples=64):
        import collections
        self.len = nsamples
        self.buffer = collections.deque(maxlen=nsamples)
        None

    def record(self,episode,verb=False):
        """ 
        given an episode (or experience more generally)
        append to bank
        """
        self.buffer.extend(episode.trajectory)
        if verb:
            print('append',self.len)
        None

    def sample(self,nsamples=1000):
        """ sample transitions from the bank
        """
        # sample list of transitions
        exp = np.random.choice(self.buffer,nsamples)
        # return dict of array
        return self.format_out(exp) 

    def format_out(self,experience):
        """ 
        experience is list of transitions
            []
        formats output as dict of nparrs
        """
        exp = {
            'st':[],'at':[],
            'rt':[],'spt':[]
        }
        # loop over transitions in trajectory
        for transition in experience:
            exp['st'].append(transition['st'])
            exp['at'].append(transition['at'])
            exp['rt'].append(transition['rt'])
            exp['spt'].append(transition['spt'])
        # list to nparr
        exp = {k:np.array(v) for k,v in exp.items()}
        return exp




class DQN(tr.nn.Module):

    def __init__(self):
        super().__init__()
        self.indim = 4 # 4 obs + 1 action
        self.stsize = 23
        self.outdim = 2 # num actions
        self.build()
        self.lossop = tr.nn.SmoothL1Loss()
        self.optiop = tr.optim.RMSprop(self.parameters(),lr=0.001)
        self.gamma=0.9

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

    def eval(self,exp):
        st = exp['st']
        at = exp['at']
        r_t = exp['rt']
        stp1 = exp['spt']
        None

    def train(self,exp):
        st = exp['st']
        at = exp['at']
        r_t = exp['rt']
        stp1 = exp['spt']
        ## form ytarget 
        # max_ap{q(s_tp1,ap_t)}
        q_stp1 = self.forward(stp1)
        q_stp1_max_ap = tr.max(q_stp1,1)[0]
        # form ytarget
        y_t = tr.Tensor(r_t) + self.gamma*q_stp1_max_ap
        y_t[-1] = r_t[-1]
        ## predict q(s_t,a_t)
        q_st = self.forward(st)
        # NB: at[:,None] unsqueezes None dim
        q_st_at = np.take_along_axis(q_st,at[:,None],axis=1)
        ## episode loss
        ep_loss = self.lossop(y_t,q_st_at.squeeze())
        # optimizer step
        self.optiop.zero_grad()
        ep_loss.backward()
        self.optiop.step()
        return np.sum(r_t)


