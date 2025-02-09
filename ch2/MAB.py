import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, K):
        self.K = K  # 老虎机数量
        self.probs = np.random.uniform(size=K)  # 每个老虎机的奖励概率
        self.best_idx = np.argmax(self.probs)  # 最优老虎机的索引
        self.best_prob = self.probs[self.best_idx]  # 最优老虎机的奖励概率

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率
        # 返回1（获奖）或0（未获奖）
        return 1 if np.random.rand() < self.probs[k] else 0


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit  # 指定多臂老虎机组合
        self.counts = np.zeros(self.bandit.K)  # 每个臂被选择的次数
        self.regret = 0  # 当前步的累计后悔
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        # 计算当前选择的动作k的遗憾值，遗憾值是指选择最佳动作的概率与实际选择动作的概率之差
        # self.bandit.best_prob 是当前环境下最佳动作的概率
        # self.bandit.probs[k] 是当前选择的动作k的概率
        # 将遗憾值累加到总的遗憾值 self.regret 中
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        # 将当前的遗憾值加入到遗憾值列表 self.regrets 中，用于后续分析
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 抛出未实现异常
        # 这个方法是一个占位符，表示在子类中需要实现具体的功能
        # 当子类没有实现这个方法时，调用这个方法会抛出NotImplementedError异常
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            # 运行一定次数,num_steps为总运行次数
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()
