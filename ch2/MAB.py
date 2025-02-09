import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, K):
        self.K = K  # �ϻ�������
        self.probs = np.random.uniform(size=K)  # ÿ���ϻ����Ľ�������
        self.best_idx = np.argmax(self.probs)  # �����ϻ���������
        self.best_prob = self.probs[self.best_idx]  # �����ϻ����Ľ�������

    def step(self, k):
        # �����ѡ����k�����˺�,�����������ϻ�����k�����˻�ý����ĸ���
        # ����1���񽱣���0��δ�񽱣�
        return 1 if np.random.rand() < self.probs[k] else 0


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit  # ָ������ϻ������
        self.counts = np.zeros(self.bandit.K)  # ÿ���۱�ѡ��Ĵ���
        self.regret = 0  # ��ǰ�����ۼƺ��
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        # ���㵱ǰѡ��Ķ���k���ź�ֵ���ź�ֵ��ָѡ����Ѷ����ĸ�����ʵ��ѡ�����ĸ���֮��
        # self.bandit.best_prob �ǵ�ǰ��������Ѷ����ĸ���
        # self.bandit.probs[k] �ǵ�ǰѡ��Ķ���k�ĸ���
        # ���ź�ֵ�ۼӵ��ܵ��ź�ֵ self.regret ��
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        # ����ǰ���ź�ֵ���뵽�ź�ֵ�б� self.regrets �У����ں�������
        self.regrets.append(self.regret)

    def run_one_step(self):
        # �׳�δʵ���쳣
        # ���������һ��ռλ������ʾ����������Ҫʵ�־���Ĺ���
        # ������û��ʵ���������ʱ����������������׳�NotImplementedError�쳣
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            # ����һ������,num_stepsΪ�����д���
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


def plot_results(solvers, solver_names):
    """�����ۻ��û���ʱ��仯��ͼ������solvers��һ���б�,�б��е�ÿ��Ԫ����һ���ض��Ĳ��ԡ�
    ��solver_namesҲ��һ���б�,�洢ÿ�����Ե�����"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()
