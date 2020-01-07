Learn RL from [Morvan](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
## 一些总结归纳
- Deep Q Learning
  - 一种td error 新估计=旧估计价值+步长x[目标-旧估计值]
  - loss:  <img src="http://latex.codecogs.com/gif.latex?\(r_j + \gamma_{max a'}Q'(\phi_{j+1},a';\theta') - Q(\phi_j,a_j;\theta))^2" />
    - 其中Q'为target network Q为eval network
    - 两个net 结构一致，每c steps将eval参数更新给target
    - 每隔c steps更新是为了固定target network参数，以防参数不固定不便于收敛。类似训练seq2seq
  - 训练网络：输入状态s，输出该状态不同动作的概率，并由概率值选取最终动作
  - 优点：
    - 相比于DP,无需一个环境模型
    - 相比于MC，无需等到一幕结束，只需下一个时刻即可。TD方法因从每次转移中学习，与采取什么后续动作无关
    - 批量蒙特卡洛方法总是找出最小化训练集上均方误差估计，而批量TD(0)总是找出完全符合马尔可夫过程模型的极大似然估计
  - 缺点：更新Q network对于连续动作不work

- Policy Gradient
  - 一种蒙特卡洛方法
    - 目的使回报期望最大  <img src="http://latex.codecogs.com/gif.latex?%5Coverline%7BR%7D_%5Ctheta%3D%5Csum_%7B%5Ctau%7DR%28%5Ctau%29P%28%5Ctau%7C%5Ctheta%29%5Capprox%5Cfrac%7B1%7D%7BN%7D%5Csum%20R%28%5Ctau%5En%29"/>
    - 最终更新函数为  ![](http://latex.codecogs.com/gif.latex?\\nabla\\overline{R}_\\theta\\approx\\frac{1}{n}\\sum_{n=1}^{N}\\sum_{t=1}^{T_n}(\\sum_{t'=t}^{T_N}\\gamma^{t'-t}r_{t'}^{n}-b)\\nabla logP(a_{t}^{n}|S_{t}^{n},\\theta))
    ![](http://latex.codecogs.com/gif.latex?\\frac{\\partial J}{\\partial \\theta_k^{(j)}}=\\sum_{i:r(i,j)=1}{\\big((\\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\\big)x_k^{(i)}}+\\lambda \\xtheta_k^{(j)})
  - 损失函数 reward更新只与t'之后相近的有关，越近越有关，越远关联性越小
    - 对于不连续动作：损失函数为收集的动作(也是网络选取的)与网络预测动作的交叉熵
    - 对于连续动作：网络输出为mu,sigma的概率分布，最大化收集动作的概率
  - 训练网络：输入状态s，输出该状态的动作概率或概率分布
  - 优点：更新动作函数，可以应对连续动作，相比于更新价值函数，收敛更快。
  - 缺点：回合更新，需要等到回合结束才能更新网络

- PPO
  - 个人理解一种多线程的policy gradient
  - 损失函数  <img src="http://latex.codecogs.com/gif.latex?J_{ppo}^{\theta^k}(\theta)\approx\sum_{s_t,a_t}min(\frac{P_\theta(a_\theta|s_t)}{P_\theta^k(a_t|s_t)}A^{\theta^k}(s_t|a_t),clip(\frac{P_\theta(a_\theta|s_t)}{P_\theta^k(a_t|s_t)},1-\xi,1+\xi)A^{\theta^k}(s_t|a_t))"/>
  - 思想：取多个worker数据来更新一个ppo
- Actor-Critic
  - 结合policy gradient和td error，保留policy gradient的动作函数，reward用actor的td error来实现
  - 损失函数：
    - critic的损失函数为最小化td error  <img src="http://latex.codecogs.com/gif.latex?\delta=R + V(s',w) - V(s,w)"/>
    - actor的损失函数与policy gradient相同，只不过把reward换成td error
  - 训练网络：
    - critic:由状态s输出价值函数v
    - actor:与policy gradient一致
  - 优点：可以更新连续动作函数，更新价值函数用来单步更新，收敛更快
- DDPG
  - Actor-Critic的改进
    - 更新actor欲使Q值最大,policy gradient -->  <img src="http://latex.codecogs.com/gif.latex?\theta^{k+1}=\theta^k + \alpha E_{s～p^{u^k}}[\nabla _\theta Q^{u^k}(s,\mu_\theta(s))]"/>  需兼顾state space和action space,每个状态可能导致不同的动作；若策略不同，导致状态不同，同<img src="http://latex.codecogs.com/gif.latex?p^{u^k}"/> 分布也不同，需大量数据收敛
    - 改进版  <img src="http://latex.codecogs.com/gif.latex?\theta^{k+1}=\theta^k + \alpha E_{s～p^{u^k}}[\nabla _\theta \mu_\theta(s) \nabla_a Q^{u^k}(s,a)| _{\mu_\theta(s)})]"/> 只需考虑state space相比上式动作确定，无需考虑策略不同导致分布变化，需更少样本
    - 结合DQN思想，actor与critic都使用两个网络一个target一个eval来加快收敛
  - 损失函数：
    - critic loss 最小化td error  <img src="http://latex.codecogs.com/gif.latex?(r_i + \gamma Q'(s_{i+1},\mu '(s_{i+1}|\theta ^{\mu '})|\theta^{Q'})-Q(s_i,a_i|\theta^Q))"/> 其中都a取自actor
    - actor梯度更新  <img src="http://latex.codecogs.com/gif.latex?\nabla _{\theta_\mu} J \approx \frac{1}{N}\sum{[\nabla _\theta \mu_\theta(s) \nabla_a Q^{u^k}(s,a)| _{\mu_\theta(s)})]}"/>
  - 训练网络：
    - actor :分target和eval两个，输入状态s输出相应动作a，更新方法与DQN相同，经过c steps两网络参数赋值
    - critic:分target和eval两个,输入状态s与取自actor的动作a,输出相应q值，更新方式同上
  - 优点：只需考虑state space训练需要更少样本，结合DQN使用两个网络可以加快收敛
- A3C
  - 个人理解一种多线程的Actor-Critic
  - 思想：每个worker的参数取自global的参数，梯度下降更新global的参数
- Sparse Reward(无法直接获取reward)
  - ICM:
    - 通过训练一个网络net输入状态s和动作a输出下一个状态s'，网络输出的误差与实际reward作为网络的reward，目的使网络探索未发现的动作
    - 改良版：上述net的输入状态s和输出状态s'经过一个特征提取网络，两个特征通过另外的网络可以得到动作a,目的是去除状态中无关信息
  - Reverse Curriculum Generation
    - 给定一个全局状态  <img src="http://latex.codecogs.com/gif.latex?s_g"/>
    - 找与  <img src="http://latex.codecogs.com/gif.latex?s_g"/> 接近的状态 <img src="http://latex.codecogs.com/gif.latex?s_1"/>
    - 记录到达  <img src="http://latex.codecogs.com/gif.latex?s_1"/> 的回报，去除回报过大(已学习过)过小(难学)的状态
    - 继续找接近的状态不断循环
  - Hierarchical Reinforement learning
    - 定小目标，一步步实现

  