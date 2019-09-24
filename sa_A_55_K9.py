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
        self.L = 54  # 客户数目，染色体长度
        self.k = 15  # 最大车的数目
        self.T = 1000 # 迭代次数
        self.capacity = 100
        self.city_demand = [0.0, 3.0, 12.0, 25.0, 4.0, 11.0, 20.0, 21.0, 10.0, 20.0, 13.0, 14.0, 16.0, 17.0, 11.0, 36.0, 6.0, 7.0, 21.0, 11.0, 17.0, 22.0, 10.0, 19.0, 21.0, 23.0, 19.0, 15.0, 22.0, 7.0, 11.0, 15.0, 22.0, 12.0, 24.0, 25.0, 2.0, 15.0, 18.0, 13.0, 3.0, 20.0, 14.0, 10.0, 10.0, 66.0, 10.0, 7.0, 12.0, 24.0, 5.0, 18.0, 7.0, 11.0, 12.0]
        self.city_distance = [[0 for i in range( self.L + 1)] for i in range( self.L + 1)]
        self.Fitness = 0
        self.oldGroup = [0 for i in range(self.L)]
        self.cat_1 = []  # 每一辆车的成本
        self.decoded = [0 for i in range(self.k)]   # 车辆安排
        self.kk = 0
        self.best_tamp = []
        self.q = 8
    # 客户间的距离
    def distance(self):
        x = [36.0, 94.0, 10.0, 16.0, 25.0, 41.0, 81.0, 14.0, 42.0, 90.0, 41.0, 21.0, 41.0, 65.0, 13.0, 21.0, 57.0, 14.0, 66.0, 58.0, 5.0, 41.0, 50.0, 84.0, 97.0, 47.0, 11.0, 60.0, 60.0, 58.0, 30.0, 9.0, 47.0, 19.0, 15.0, 88.0, 33.0, 21.0, 57.0, 81.0, 49.0, 51.0, 9.0, 84.0, 95.0, 89.0, 10.0, 69.0, 75.0, 97.0, 74.0, 1.0, 96.0, 46.0, 74.0]
        y = [64.0, 47.0, 23.0, 46.0, 79.0, 30.0, 45.0, 79.0, 56.0, 17.0, 39.0, 14.0, 46.0, 96.0, 49.0, 14.0, 2.0, 42.0, 62.0, 96.0, 51.0, 50.0, 99.0, 85.0, 90.0, 76.0, 54.0, 97.0, 89.0, 68.0, 93.0, 60.0, 44.0, 40.0, 40.0, 21.0, 58.0, 51.0, 7.0, 6.0, 6.0, 78.0, 62.0, 36.0, 76.0, 44.0, 49.0, 16.0, 66.0, 11.0, 69.0, 14.0, 91.0, 22.0, 92.0]
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
                evaluation = evaluation + cur_d - self.city_distance[Gh[j-1]][Gh[j]] + self.city_distance[Gh[j -1]][0]
                self.cat_1.append(cat_cost)  # 每一辆车的成本
                cur_d = self.city_distance[0][Gh[j]]  # 从配送中心到当前客户j的距离
                cur_q = self.city_demand[Gh[j]]
            else:
                self.decoded[i] += 1
        # 加上最后一辆车走的距离
        self.cat_1.append(cur_d + self.city_distance[Gh[self.L - 1]][Gh[0]])
        evaluation = evaluation + cur_d + self.city_distance[Gh[self.L - 1]][0]
        self.kk = i
    # 看车辆使用数目是否大于规定的车数，最多超出1辆
        flag = i - self.q
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
                evaluation = evaluation + cur_d - self.city_distance[Gh[j-1]][Gh[j]] + self.city_distance[Gh[j -1]][0]
                cur_d = self.city_distance[0][Gh[j]]  # 从配送中心到当前客户j的距离
                cur_q = self.city_demand[Gh[j]]
            else:
                self.decoded[i] += 1
        # 加上最后一辆车走的距离
        # evaluation = evaluation + cur_d
        decodedEvaluation = evaluation + cur_d + self.city_distance[Gh[self.L - 1]][0]
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
        t = (1, 100)
        markovlen = 60
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
        self.Evaluate(Gh)
        for m in range(0, self.kk+1):  # 每条路径中进行调换
            Gh_1 = Gh.copy()
            A_gh = [(0) for i in range(self.decoded[m] + 2)]
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
        for i in range(self.T):
            self.SA(self.oldGroup)
            self.Alns_destory(self.oldGroup)
            self.decoding(self.oldGroup)
        time_end = time.time()
        self.plot(self.best_tamp)
        print('程序运行时间', time_end - time_start)

if __name__ == '__main__':
    ga = Gh()
    ga.main()
