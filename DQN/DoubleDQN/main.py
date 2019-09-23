import gym
from RL_brain import DoubleDQN
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE,n_features=3,memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001,double_q=False,sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE,n_features=3,memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001,double_q=True,sess=sess,output_graph=True
    )

    sess.run(tf.global_variables_initializer())
def train(RL):
    step = 0
    obervation = env.reset()
    while True:
        if step - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(obervation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
        obervation_,reward,done,info = env.step(np.array([f_action]))

        reward /= 10
        RL.store_transition(obervation,action,reward,obervation_)

    
        if step > MEMORY_SIZE:

            RL.learn()

        if step - MEMORY_SIZE > 20000:
            break 

        
        obervation = obervation_
        
       
        step += 1
    return RL.cost_his


q_natural = train(natural_DQN)
q_double = train(double_DQN)

plt.plot(np.array(q_natural),c='r',label='natural')
plt.plot(np.array(q_double),c='b',label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training step')
plt.grid()
plt.show()