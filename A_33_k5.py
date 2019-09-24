import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import copy
import time
# 初始化函数
class Gh(object):
    def __init__(self):
        self.bestEvaluation = 0
        self.pw = 1500  # 车辆超额出发权重
        self.L = 32  # 客户数目，染色体长度
        self.k = 15  # 最大车的数目
        self.T = 100# 迭代次数
        self.capacity = 100
        self.city_demand = [0.0, 5.0, 23.0, 14.0, 13.0, 8.0, 18.0, 19.0, 10.0, 18.0, 20.0, 5.0, 9.0, 23.0, 9.0, 18.0, 10.0, 24.0, 13.0, 14.0, 8.0, 10.0, 19.0, 14.0, 13.0, 14.0, 2.0, 23.0, 15.0, 8.0, 20.0, 24.0, 3.0]
        self.city_distance = [[0 for i in range( self.L + 1)] for i in range( self.L + 1)]
        self.Fitness = 0
        self.oldGroup = [0 for i in range(self.L)]
        self.cat_1 = []  # 每一辆车的成本
        self.decoded = [0 for i in range(self.k)]   # 车辆安排
        self.kk = 0
        self.best_tamp = []
        self.q = 4
    # 客户间的距离
    def distance(self):
        x = [42.0, 77.0, 28.0, 77.0, 32.0, 32.0, 42.0, 8.0, 7.0, 82.0, 48.0, 53.0, 39.0, 7.0, 67.0, 54.0, 72.0, 73.0, 59.0, 58.0, 23.0, 68.0, 47.0, 52.0, 32.0, 39.0, 17.0, 38.0, 58.0, 82.0, 42.0, 68.0, 7.0]
        y = [68.0, 97.0, 64.0, 39.0, 33.0, 8.0, 92.0, 3.0, 14.0, 17.0, 13.0, 82.0, 27.0, 24.0, 98.0, 52.0, 43.0, 3.0, 77.0, 97.0, 43.0, 98.0, 62.0, 72.0, 88.0, 7.0, 8.0, 7.0, 74.0, 67.0, 7.0, 82.0, 48.0]
        for i in range(0,  self.L + 1):
            distance1 = []
            for j in range(0,  self.L + 1):
                distance0 = ((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j])) ** 0.5
                distance1.append(distance0)
            self.city_distance[i] = distance1

    # 染色体评价函数，输入一个染色体，得到该染色体评价值
    def Evaluate(self, Gh):
        cur_d = self.city_distance[0][Gh[0]]
        cur_q = self.city_demand[Gh[0]]
        self.cat_1 = []
        self.kk = 0
        self.decoded = [0 for i in range(self.k)]
        i = 0  # 从1号车开始，默认第一辆车能满足第一个客户的需求
        self.decoded[i] = 1
        evaluation = 0  # 评价值初始值为0
        for j in range(1,  self.L):
            cur_q = cur_q + self.city_demand[Gh[j]]
            cur_d = cur_d + self.city_distance[Gh[j-1]][Gh[j]]
            # 如果当前客户需求大于车辆的最大载重调用下一辆车
            if cur_q >= self.capacity:
                i += 1
                self.decoded[i] = 1
                cat_cost = cur_d - self.city_distance[Gh[j]][Gh[j - 1]]
                evaluation = evaluation + cur_d - self.city_distance[Gh[j-1]][Gh[j]]
                self.cat_1.append(cat_cost)  # 每一辆车的成本
                cur_d = self.city_distance[0][Gh[j]]  # 从配送中心到当前客户j的距离
                cur_q = self.city_demand[Gh[j]]
            else:
                self.decoded[i] += 1
        # 加上最后一辆车走的距离
        self.cat_1.append(cur_d)
        evaluation = evaluation + cur_d
        self.kk = i
    # 看车辆使用数目是否大于规定的车数，最多超出1辆
        flag = i - 4
        if flag < 0:
            flag = 0
        evaluation = evaluation + flag * self.pw  # 超额车辆数的惩罚权重
        return evaluation  # 压缩权重值

    # 染色体解码函数，输入一个染色体，得到该染色体表达的每辆车的服务的客户顺序
    def decoding(self, Gh):
        cur_d = self.city_distance[0][Gh[0]]  # 第一个是参数是染色体序号
        cur_q = self.city_demand[Gh[0]]
        i = 0  # 从1号车开始，默认第一辆车能满足第一个客户的需求
        self.decoded = [0 for i in range(self.k)]
        self.decoded[i] = 1
        evaluation = 0  # 评价值初始值为0
        for j in range(1,  self.L):
            cur_q = cur_q + self.city_demand[Gh[j]]
            cur_d = cur_d + self.city_distance[Gh[j-1]][Gh[j]]
            # 如果当前客户需求大于车辆的最大载重调用下一辆车
            if cur_q >= self.capacity:
                i += 1
                self.decoded[i] = 1
                evaluation = evaluation + cur_d - self.city_distance[Gh[j-1]][Gh[j]]
                cur_d = self.city_distance[0][Gh[j]]  # 从配送中心到当前客户j的距离
                cur_q = self.city_demand[Gh[j]]
            else:
                self.decoded[i] += 1
        # 加上最后一辆车走的距离
        # evaluation = evaluation + cur_d
        decodedEvaluation = evaluation + cur_d
        kk = i
        self.best_tamp.append(decodedEvaluation)
        print(decodedEvaluation, kk, self.decoded, Gh)

    # 初始化解
    def initGroup(self):
            self.oldGroup = random.sample(range(1, self.L + 1), self.L)

    def ftiness(self):
            self.Fitness = self.Evaluate(self.oldGroup)

    def initpara(self):
        alpha = 0.99
        t = (1, 5)
        markovlen = 30
        return alpha, t, markovlen
    # 自适应大领域算法。
    def SA(self, Gh):
        solutionnew = copy.deepcopy(Gh)
        solutioncurrent = solutionnew.copy()  # 当前的解决方案
        valuecurrent = self.Evaluate(solutioncurrent)
        solutionbest = solutionnew.copy()
        valuebest = self.Evaluate(solutioncurrent)
        alpha, t2, markovlen = self.initpara()
        t = t2[1]
        while t > t2[0]:
            for i in np.arange(markovlen):
                #  markovlen 迭代次数
                # 下面的两交换和三角换是两种扰动方式，用于产生新解
                if np.random.rand() > 0.6:  # 交换路径中的这2个节点的顺序
                    # np.random.rand()产生[0, 1)区间的均匀随机数
                    while True:  # 产生两个不同的随机数
                        loc1 = np.int(np.ceil(np.random.rand() * (self.L - 1)))
                        loc2 = np.int(np.ceil(np.random.rand() * (self.L - 1)))
                        if loc1 != loc2:
                            break
                    solutionnew[loc1], solutionnew[loc2] = solutionnew[loc2], solutionnew[loc1]
                else:  # 三交换
                    while True:
                        loc1 = np.int(np.ceil(np.random.rand() * (self.L - 1)))
                        loc2 = np.int(np.ceil(np.random.rand() * (self.L - 1)))
                        loc3 = np.int(np.ceil(np.random.rand() * (self.L - 1)))

                        if ((loc1 != loc2) & (loc2 != loc3) & (loc1 != loc3)):
                            break

                    # 下面的三个判断语句使得loc1<loc2<loc3
                    if loc1 > loc2:
                        loc1, loc2 = loc2, loc1
                    if loc2 > loc3:
                        loc2, loc3 = loc3, loc2
                    if loc1 > loc2:
                        loc1, loc2 = loc2, loc1

                    # 下面的三行代码将[loc1,loc2)区间的数据插入到loc3之后
                    tmplist = solutionnew[loc1:loc2].copy()
                    solutionnew[loc1:loc3 - loc2 + 1 + loc1] = solutionnew[loc2:loc3 + 1].copy()
                    solutionnew[loc3 - loc2 + 1 + loc1:loc3 + 1] = tmplist.copy()

                valuenew = self.Evaluate(solutionnew)
                if valuenew < valuecurrent:  # 接受该解
                    # 更新solutioncurrent 和solutionbest
                    valuecurrent = valuenew
                    solutioncurrent = solutionnew.copy()

                    if valuenew < valuebest:
                        valuebest = valuenew
                        solutionbest = solutionnew.copy()
                else:  # 按一定的概率接受该解
                    if np.random.rand() < np.exp(-(valuenew - valuecurrent) / t):
                        valuecurrent = valuenew
                        solutioncurrent = solutionnew.copy()
                    else:
                        solutionnew = solutioncurrent.copy()
            t = alpha * t
        self.best_tamp.append(valuebest)
        for i in range(self.L):
            Gh[i] = solutionbest[i]

     # 自适应大领域算法。
    def Alns_destory(self, Gh):
        sum = 0
        cost_all = 0
        for m in range(0, self.kk):  # 每条路径中进行调换
            Gh_1 = Gh.copy()
            A_gh = [(0) for i in range(self.decoded[m] + 1)]
            sum += self.decoded[m - 1]
            for i in range(0, self.decoded[m]):
                A_gh[i + 1] = Gh[sum + i]
            for i in range(self.L):  # 2_opt重新定位次数
                if len(A_gh) == 1:
                    continue
                else:
                    while True:
                        ran1 = random.randint(1, len(A_gh) - 2)
                        ran2 = random.randint(1, len(A_gh) - 2)
                        if ran1 != ran2:
                            break
                    cost_ran1 = self.city_distance[A_gh[ran1 - 1]][A_gh[ran1]] + self.city_distance[A_gh[ran1]][
                        A_gh[ran1 + 1]]
                    cost_ran2 = self.city_distance[A_gh[ran2 - 1]][A_gh[ran2]] + self.city_distance[A_gh[ran2]][
                        A_gh[ran2 + 1]]
                    cost_new_ran2 = self.city_distance[A_gh[ran1 - 1]][A_gh[ran2]] + self.city_distance[A_gh[ran2]][
                        A_gh[ran1 + 1]]
                    cost_new_ran1 = self.city_distance[A_gh[ran2 - 1]][A_gh[ran1]] + self.city_distance[A_gh[ran1]][
                        A_gh[ran2 + 1]]
                    a = cost_ran1 + cost_ran2
                    b = cost_new_ran1 + cost_new_ran2
                    if a > b:
                        A_gh[ran1], A_gh[ran2] = A_gh[ran2], A_gh[ran1]
            for i in range(self.decoded[m]):
                Gh_1[sum + i] = A_gh[i + 1]
            if self.Evaluate(Gh) > self.Evaluate(Gh_1):
                for i in range(self.L):
                    Gh[i] = Gh_1[i]

    # 抽取种群中的个体进行局部
    def plot(self, best_gh):
        n = []
        m = []
        for i in range(len(best_gh)):
            n.append(i)
            m.append(best_gh[i])
        plt.plot(n, m)
        plt.show()

    def main(self):
        self.distance()
        self.initGroup()
        time_start = time.time()
        for t in range(self.T):
            self.SA(self.oldGroup)
            # self.Alns_destory(self.oldGroup)
            self.decoding(self.oldGroup)
        time_end = time.time()
        self.plot(self.best_tamp)
        print('程序运行时间', time_end - time_start)

if __name__ == '__main__':
    ga = Gh()
    ga.main()
