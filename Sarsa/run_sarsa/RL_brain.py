import numpy as np 
import pandas as pd 

class SarsaTable(object):
    def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    
    def choose_action(self,observation):
        self.check_state_exist(observation)
        if np.random.rand()<self.epsilon:
            state_action = self.q_table.loc[observation,:]
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_,a_]
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)