import collections
import numpy
import gym
from collections import namedtuple
import torch as tr
from torch.distributions.categorical import Categorical
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

    def play_ep(self,pi=None):
        """ 
        given policy, return trajectory
            pi(s_t) -> a_t
        returns episode_exp = {st:[],at:[],rt:[],spt:[]}
        """        
        # config env
        self.env.reset()
        if type(pi)==type(None):
            pi = self.rand_policy
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
            a_t = pi(s_t)
            env_out_t = self.env.step(np.array(a_t))
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




def compute_returns(rewards,gamma=1.0):
    """ 
    given rewards, compute discounted return
    G_t = sum_k [g^k * r(t+k)]; k=0...T-t
    """ 
    T = len(rewards) 
    returns = np.array([
        np.sum(np.array(
            rewards[t:])*np.array(
            [gamma**i for i in range(T-t)]
        )) for t in range(T)
    ])
    return returns


def unpack_expL(expLoD):
    """ 
    given list of experience (namedtups)
        expLoD [{t,s,a,r,sp}_t]
    return dict of np.arrs 
        exp {s:[],a:[],r:[],sp:[]}
    """
    expDoL = Experience(*zip(*expLoD))._asdict()
    return {k:np.array(v) for k,v in expDoL.items()}



class REINFORCE(tr.nn.Module):
  
    def __init__(self,indim=4,nactions=2,stsize=18,learnrate=0.005):
        super().__init__()
        self.indim = indim
        self.stsize = stsize
        self.nactions = nactions
        self.learnrate = learnrate
        self.build1()
        return None
  
    def build1(self):
        # policy parameters
        self.in2hid = tr.nn.Linear(self.indim,self.stsize,bias=True)
        self.hid2hid = tr.nn.Linear(self.stsize,self.stsize,bias=True)
        self.hid2val = tr.nn.Linear(self.stsize,1,bias=True)
        self.hid2pi = tr.nn.Linear(self.stsize,self.nactions,bias=True)
        # optimization
        self.optiop = tr.optim.RMSprop(self.parameters(), 
          lr=self.learnrate
        )
        return None
    
    def forward(self,xin):
        """ 
        xin [batch,dim]
        returns activations of output layer
            vhat: [batch,1]
            phat: [batch,nactions]
        """
        xin = tr.Tensor(xin)
        hact = self.in2hid(xin).relu()
        # hact = self.hid2hid(hact).relu()
        vhat = self.hid2val(hact)
        pact = self.hid2pi(hact)
        return vhat,pact

    def act(self,xin):
        """ 
        xin [batch,stimdim]
        used to interact with environment
        forward prop, then apply policy
        returns action [batch,1]
        """
        vhat,pact = self.forward(xin)
        # pism = pact.softmax(-1)
        # if pism.min()<0.05:
        #     pism = tr.Tensor([0.05,0.95])
        # pidistr = Categorical(pism)

        # actions = pidistr.sample()
        if np.random.random() > 0.9:
            return np.random.randint(2)
        actions = pact.softmax(-1).argmax()
        return actions

    def eval(self,expD):
        """ """
        data = {}
        ## entropy
        vhat,pact = self.forward(expD['state'])
        pra = pact.softmax(-1)
        entropy = -1 * tr.sum(pra*pra.log2(),-1).mean()
        data['entropy'] = entropy.detach().numpy()
        ## value
        returns = compute_returns(expD['reward']) 
        data['delta'] = np.mean(returns - vhat.detach().numpy())
        return data

    def update(self,expD):
        """ given dictionary of experience
            expD = {'reward':[tsteps],'state':[tsteps],...}
        """
        # assuming exp_dict is temporal:
        returns = compute_returns(expD['reward'],gamma=0.95) 
        states,actions = expD['state'],tr.Tensor(expD['action'])
        vhat,pact = self.forward(expD['state'])
        los = 0
        for vh,pa,At,Gt in zip(vhat,pact,actions,returns):
            delta = Gt - vh
            los_pi = delta*tr.log(pa.softmax(-1)[At.numpy()])
            los_val = tr.square(vh - Gt)
            los += los_val-los_pi
        # update step
        self.optiop.zero_grad()
        los.backward()
        self.optiop.step()
            
        return None 
  
  



