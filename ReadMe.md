Learn RL from [Morvan](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
## some trick
- q learning --> 一种td error 学习方法，通过更新价值函数来选取动作
- policy gradient --> 一种蒙特卡洛方法
- - 优点：更新动作函数，可以应对连续动作，相比于更新价值函数，收敛更快。
- - 缺点：回合更新
- actor - critic --> 结合policy gradient和td error，既可以更新连续动作函数，更新价值函数用来单步更新，收敛更快。
