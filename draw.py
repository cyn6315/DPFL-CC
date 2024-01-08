import matplotlib.pyplot as plt
import numpy as np

def drawDPCluster():
    plt.switch_backend('agg')
    plt.clf()
    plt.rcParams['font.family'] = 'SimSun'
    accuracyList=np.array([0.5526, 0.5683,  0.6093, 0.5906, 0.5870])
    plt.plot(range(2,7,1), accuracyList, color='tomato', linewidth=1.5, label='Personal DPFL-CC')

    accuracyList=np.array([0.3189, 0.4218, 0.4646, 0.5024, 0.5217])
    plt.plot(range(2,7,1), accuracyList, color='gold', linewidth=1.5, label='DPFL-CC')
    plt.xticks(np.arange(2,7,1))
    plt.title("个性化 VS 非个性化 (ε=8, δ=1e-4)")
    plt.xlabel('单个客户端数据的类别个数')
    plt.ylabel('ACC')
    plt.legend()
    plt.grid()
    plt.savefig("./personalization.jpg")

if __name__ == '__main__':
    # drawNoclusterVScluster()
    drawDPCluster()