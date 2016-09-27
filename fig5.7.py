# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 00:04:59 2016

@author: Liao Fangzhou
"""
import matplotlib.pyplot as plt
import numpy as np
class environment():
    def __init__(self):
        self.playercard = [1,2]
        self.dealercard = [2,self.getcard()]
        self.state_all= [self.playercard,self.dealercard]
        self.turn = 0 #0 for player, 1 for dealer
    def getstate(self,agentid):
        oppostate = self.state_all[1-agentid]
        summing,usableace = self.count(agentid)
        state = [summing,oppostate[0],usableace]
        return state
#        return 
    def move(self,act,agentid):
        mystate = self.state_all[agentid]
        end = False
        r = 0
        if act =='hit':
            mystate.append(self.getcard())
            s0,_ = self.count(agentid)
            if s0>21:
                if agentid==0:
                    r = -1
                else: 
                    r = 1
                end = True
        else: 
            self.turn = 1-self.turn
            if agentid==1:
                s0,_ = self.count(0)
                s1,_ = self.count(1)
                if s0==s1:
                    r=0
                elif s0>s1:
                    r=1
                elif s0<s1:
                    r=-1
                end=True
        return r,end,self.getstate(agentid)
#        s1,_ = self.count(1)
    def count(self,agentid):
        mystate = self.state_all[agentid]
        if 1 in mystate:
            usableace =1
        else:
            usableace = 0
        summing = np.sum(mystate)
        if summing<=11 and usableace==1:
            summing = summing+10
        elif summing>11:
            usableace=0
        
        
        return summing,usableace
        
    def getcard(self):
        x = np.random.randint(low=1,high=14)
        if x>10:
            x=10
        return x
        
class agent():
    def __init__(self,policy,learnMethod='Markov',epsilon =0,stickpoint = 0):
        self.policy=policy
        self.possibleact = ['hit','stick']
        self.epsilon = epsilon
        self.valueFunction = list(np.zeros(11*10*2))
#        if policy=='greedy':
#            self.epsilon = epsilon
        if policy == 'fixed':
            self.policy=policy
            self.stickpoint = stickpoint
        elif learnMethod == 'Markov':
            None
    def act(self,state):
        if self.policy=='random':
            tmp = np.random.randint(low=0,high=2)
            return self.possibleact[tmp],[0.5,0.5]
        elif self.policy=='fixed':
            if state[0]>=self.stickpoint:
                return 'stick',[0,1]
            else:
                return 'hit',[1,0]

    def learn_markov_off(self,history,reward,sample='ordi'):
        rho = 1
        startState = history[0][0]
        for h in history[::-1]:
            state,act,p = h
            myact,myp = self.act(translatestate(h[0]))
#            myact = self.possibleact.index(myact)
            rho = myp[act]/p[act]*rho
        if not hasattr(self,'recordN'):
            self.recordN = list(np.zeros_like(self.valueFunction))
            self.sumrho = 0.
        self.recordN[startState]=self.recordN[startState]+1
        V = self.valueFunction[startState] 
        if sample=='ordi':
            self.valueFunction[startState] = V+1./self.recordN[startState]*rho*(reward-V)
        elif sample =='importance':
            if rho>0:
                self.valueFunction[startState] = V*self.sumrho/(self.sumrho+rho)+reward*rho/(self.sumrho+rho)
            self.sumrho = self.sumrho+rho
        return self.valueFunction[startState]
    def learnFromAction(self):
        None
        
def translatestate(state):
    statespace=[11,10,2]
    
    if type(state)==int:
        newstate = list(np.unravel_index(state,statespace))
        newstate[0] = newstate[0]+11
        newstate[1] = newstate[1]+1
    elif type(state)==list and len(state)==3:
        tmp = state[:]
        tmp[0] = tmp[0]-11
        tmp[1] = tmp[1]-1
        newstate = int(np.ravel_multi_index(tmp,statespace))
    return newstate

class runner():
    def __init__(self):
#        self.env = environment()
        self.player = agent(policy='fixed',stickpoint=20)
        self.player_behave = agent(policy='random')
        self.dealer = agent(policy='fixed',stickpoint=17)
        self.allhistory = []
        self.allrewardhistory = []
    def runEpisode(self,player,dealer,record = False):
        env = environment()
        end = False
        agentlist = [player,dealer]
        playerhistory = []
        while not end:
            turn = env.turn
            s = env.getstate(turn)
            act,p = agentlist[turn].act(s)
            r,end,s2 = env.move(act,turn)
#            print [s,act,s2,r,turn]
            if turn==0:
                playerhistory.append([translatestate(s),player.possibleact.index(act),p])
        playerhistory = playerhistory
        if record:
            self.allhistory.append( playerhistory)
            self.allrewardhistory.append(r)
        return [playerhistory,r]
#en = environment()
#for i in range(1000000):
#    r.runEpisode(r.player,r.dealer,record=True)
#Gtrue = np.mean(r.allrewardhistory)
Gtrue = -0.2768
learnlog=np.zeros([100,1000])
for sim in range(100):
    run = runner()

    for i in range(1000):
        [hist,rew] = run.runEpisode(run.player_behave,run.dealer)
        V= run.player.learn_markov_off(hist,rew,sample='importance')
        learnlog[sim,i]=V

learnlog2=np.zeros([100,1000])
for sim in range(100):
    run = runner()

    for i in range(1000):
        [hist,rew] = run.runEpisode(run.player_behave,run.dealer)
        V= run.player.learn_markov_off(hist,rew,sample='ordi')
        learnlog2[sim,i]=V

plt.plot(np.mean(np.square(learnlog-Gtrue),axis=0))
plt.plot(np.mean(np.square(learnlog2-Gtrue),axis=0))
plt.ylim([0,4])
plt.xscale('log')
plt.legend(['importance','ordinary'])
#print en.getstate(1)