import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def drawDPCluster():
    plt.switch_backend('agg')
    plt.clf()
    plt.rcParams['font.family'] = 'SimHei'
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

def drawEpsilon():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)
    accuracyList=np.array([ 0.4015, 0.5484,  0.5615, 0.5712, 0.5686, 0.5766])
    plt.plot(range(0,51,10), accuracyList, color='gold', linewidth=1.5, marker='.', label='ε=2')

    accuracyList=np.array([ 0.4015,0.5579,  0.5739, 0.5885, 0.5930, 0.5951])
    plt.plot(range(0,51,10), accuracyList, color='#ffb6c1', linewidth=1.5, marker='x', label='ε=4')

    accuracyList=np.array([ 0.4015,0.5675,  0.5862, 0.5946, 0.5982, 0.6027])
    plt.plot(range(0,51,10), accuracyList, color='#90ee90', linewidth=1.5,marker='s', label='ε=6')

    accuracyList=np.array([ 0.4015,0.5711,  0.5933, 0.6010, 0.6082, 0.6174])
    plt.plot(range(0,51,10), accuracyList, color='lightskyblue', linewidth=1.5, marker='^', label='ε=8')
    plt.xticks(np.arange(0,51,10))
    plt.xlabel('联邦学习轮数',fontproperties=myfont)
    plt.ylabel('ACC')
    plt.legend()
    plt.grid()
    plt.savefig("./drawEpsilon.jpg")


def drawR():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)

    accuracyList=np.array([ 0.4015, 0.5685, 0.5903, 0.5974, 0.6011, 0.6087])
    plt.plot(range(0,51,10), accuracyList, color='gold', linewidth=1.5, marker='.', label='r=2')

    accuracyList=np.array([ 0.4015,0.5729,  0.5943, 0.6018, 0.6082, 0.6174])
    plt.plot(range(0,51,10), accuracyList, color='#ffb6c1', linewidth=1.5, marker='x', label='r=4')

    accuracyList=np.array([ 0.4015,0.5651,  0.5840, 0.5939, 0.5965, 0.6002])
    plt.plot(range(0,51,10), accuracyList, color='turquoise', linewidth=1.5,marker='s', label='r=6')
    
    accuracyList=np.array([ 0.4015,0.5584,  0.5790, 0.5896, 0.5930, 0.5949])
    plt.plot(range(0,51,10), accuracyList, color='lightskyblue', linewidth=1.5, marker='^', label='r=8')
    plt.xticks(np.arange(0,51,10))
    plt.xlabel('联邦学习轮数',fontproperties=myfont, fontsize=12)
    plt.ylabel('ACC', fontsize=12)
    plt.legend()
    plt.grid()
    plt.savefig("./drawR.jpg")




def drawShoulian():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)
    accuracyList=np.array([ 0.1739, 0.4920, 0.4431, 0.4424, 0.4423, 0.4548])
    plt.plot(range(0,51,10), accuracyList, color='#ffb6c1', linewidth=1.5, marker='x', label='DP-FLDC')

    accuracyList=np.array([ 0.1739, 0.5006, 0.4786, 0.4668, 0.4684, 0.4696])
    plt.plot(range(0,51,10), accuracyList, color='turquoise', linewidth=1.5,marker='s', label='FedProx')
    
    accuracyList=np.array([ 0.1739, 0.4936, 0.4774, 0.4489, 0.4597, 0.4679])
    plt.plot(range(0,51,10), accuracyList, color='lightskyblue', linewidth=1.5, marker='^', label='PrivateFL')

    accuracyList=np.array([ 0.1739, 0.6774, 0.6980, 0.7047, 0.7044, 0.7204])
    plt.plot(range(0,51,10), accuracyList, color='gold', linewidth=1.5, marker='.', label='DP-PFLDC-Reassign')
    plt.xticks(np.arange(0,51,10))

    # accuracyList=np.array([ 0.4015, 0.4930, 0.4683, 0.4485,  0.4249, 0.4168, 0.4138])
    # plt.plot(range(0,31,5), accuracyList, color='#ffb6c1', linewidth=1.5, marker='x', label='DP-FLDC')

    # accuracyList=np.array([ 0.4015, 0.4630,  0.4721, 0.4882,  0.4925, 0.4989, 0.5031])
    # plt.plot(range(0,31,5), accuracyList, color='turquoise', linewidth=1.5,marker='s', label='FedProx')
    
    # accuracyList=np.array([ 0.4015, 0.4822, 0.4783, 0.4721, 0.4558, 0.4331, 0.4275])
    # plt.plot(range(0,31,5), accuracyList, color='lightskyblue', linewidth=1.5, marker='^', label='FedBN')

    # accuracyList=np.array([ 0.4015, 0.5542, 0.585, 0.5896, 0.5904, 0.6056, 0.6002])
    # plt.plot(range(0,31,5), accuracyList, color='gold', linewidth=1.5, marker='.', label='DP-PFLDC-Reassign')
    # plt.xticks(np.arange(0,31,5))
    plt.xlabel('联邦学习轮数',fontproperties=myfont, fontsize=12)
    plt.ylabel('ACC', fontsize=12)
    plt.legend()
    plt.grid()
    plt.savefig("./drawShoulian.jpg")

def drawAlpha():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)

    accuracyList=np.array([ 0.4015, 0.585, 0.5882, 0.595, 0.586, 0.592])
    plt.plot(range(0,51,10), accuracyList, color='lightskyblue', linewidth=1.5, marker='.', label='α=0.05')

    accuracyList=np.array([ 0.4015, 0.580, 0.5904, 0.6002, 0.5964, 0.6056])
    plt.plot(range(0,51,10), accuracyList, color='#ffb6c1', linewidth=1.5, marker='x', label='α=0.2')

    accuracyList=np.array([ 0.4015, 0.558, 0.572, 0.579, 0.585,  0.586])
    plt.plot(range(0,51,10), accuracyList, color='gold', linewidth=1.5, marker='^', label='α=0.8')

    plt.xticks(range(0,51,10))

    # accuracyList=np.array([ 0.4015, 0.5670, 0.5679, 0.582, 0.6000, 0.592, 0.589])
    # plt.plot(range(0,31,5), accuracyList, color='lightskyblue', linewidth=1.5, marker='.', label='α=0.05')

    # accuracyList=np.array([ 0.4015, 0.5630, 0.585, 0.5896, 0.5904, 0.6056, 0.6002])
    # plt.plot(range(0,31,5), accuracyList, color='#ffb6c1', linewidth=1.5, marker='x', label='α=0.2')

    # accuracyList=np.array([ 0.4015, 0.554,  0.5716, 0.5760, 0.579, 0.584,  0.586])
    # plt.plot(range(0,31,5), accuracyList, color='gold', linewidth=1.5, marker='^', label='α=0.8')

    # plt.xticks(range(0,31,5))
    plt.xlabel('联邦学习轮数',fontproperties=myfont, fontsize=12)
    plt.ylabel('ACC', fontsize=12)
    plt.legend()
    plt.grid()
    plt.savefig("./drawAlpha.jpg")


def drawM():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)

    accuracyList=np.array([0.2648, 0.3329, 0.3860, 0.4320, 0.4682])
    plt.plot(range(2,7,1), accuracyList, color='#ffb6c1', linewidth=1.5, marker='x', label='DP-FLDC')

    accuracyList=np.array([ 0.5526, 0.5683, 0.6056, 0.5906,  0.5870])
    plt.plot(range(2,7,1), accuracyList, color='lightskyblue', linewidth=1.5, marker='.', label='DP-PFLDC-Reassign')
    plt.xticks(np.arange(2,7,1))
    plt.xlabel('客户端数据类别数M',fontproperties=myfont, fontsize=12)
    plt.ylabel('ACC', fontsize=12)
    plt.legend()
    plt.grid()
    plt.savefig("./drawM.jpg")


def drawShoulian100():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)

    accuracyList=np.array([ 0.1257, 0.1892, 0.2154, 0.2108, 0.2163, 0.2167])
    plt.plot(range(0,51,10), accuracyList, color='#ffb6c1', linewidth=1.5, marker='x', label='DP-FLDC')

    accuracyList=np.array([ 0.1257, 0.2053, 0.2140, 0.2216, 0.2237, 0.2286])
    plt.plot(range(0,51,10), accuracyList, color='turquoise', linewidth=1.5,marker='s', label='FedProx')

    accuracyList=np.array([ 0.1257, 0.2013, 0.2083, 0.2151, 0.2213, 0.2205])
    plt.plot(range(0,51,10), accuracyList, color='lightskyblue', linewidth=1.5, marker='^', label='PrivateFL')

    accuracyList=np.array([ 0.1257, 0.2439, 0.2655, 0.2718, 0.2819, 0.2927])
    plt.plot(range(0,51,10), accuracyList, color='gold', linewidth=1.5, marker='.', label='DP-PFLDC-Reassign')
    plt.xticks(np.arange(0,51,10))

    plt.xlabel('联邦学习轮数',fontproperties=myfont, fontsize=12)
    plt.ylabel('ACC', fontsize=12)
    plt.legend()
    plt.grid()
    plt.savefig("./drawShoulian100.jpg")


def draw_epsilon_zhuzhuang():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)

    totalWidth=0.8
    labelNums=3 
    barWidth=totalWidth/(labelNums)
    seriesNums=4 

    ARI_x=[x for x in range(seriesNums)]
    NMI_x=[x+barWidth for x in range(seriesNums)]
    ACC_x=[x+2*barWidth for x in range(seriesNums)]

    ARI_y=[0.3532, 0.3774, 0.3838, 0.3993]
    NMI_y=[0.4283, 0.4512,0.4582,0.4698]
    ACC_y=[0.5746, 0.5953, 0.6007, 0.6174]

    plt.bar(ARI_x, height=ARI_y, label="ARI", width=barWidth, color='white', edgecolor='#61355f',hatch="/", linewidth=2)
    plt.bar(NMI_x, height=NMI_y, label="NMI", width=barWidth, color='white', edgecolor='#b58fb5', linewidth=2)
    plt.bar(ACC_x, height=ACC_y, label="ACC", width=barWidth, color='white',edgecolor='#3c6388', hatch="//", linewidth=2)

    cout=0
    for x1, yy in zip(ARI_x, ARI_y):
        cout+=1
        if cout==4:
            plt.text(x1, yy+0.001, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)
    cout=0
    for x1, yy in zip(NMI_x, NMI_y):
        cout+=1
        if cout==4:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)
    cout=0
    for x1, yy in zip(ACC_x, ACC_y):
        cout+=1
        if cout==4:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)

    plt.xticks([x+barWidth/2*(labelNums-1) for x in range(seriesNums)], ["ε=2","ε=4","ε=6","ε=8"])
    plt.xlabel("隐私预算ε",fontproperties=myfont, fontsize=12)
    plt.ylabel("聚类性能",fontproperties=myfont, fontsize=12)
    plt.legend(loc = (0, 0.8))
    plt.savefig("./draw_epsilon_zhuzhuang.png", dpi=300)


def draw_r_zhuzhuang():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)

    totalWidth=0.8
    labelNums=3 
    barWidth=totalWidth/labelNums 
    seriesNums=4 

    ARI_x=[x for x in range(seriesNums)]
    NMI_x=[x+barWidth for x in range(seriesNums)]
    ACC_x=[x+2*barWidth for x in range(seriesNums)]

    ARI_y=[0.3937, 0.3993, 0.3873, 0.3843]
    NMI_y=[0.4678, 0.4698, 0.4592, 0.4590]
    ACC_y=[0.6087, 0.6174, 0.6026, 0.6009]

    plt.bar(ARI_x, height=ARI_y, label="ARI", width=barWidth, color='white', edgecolor='#D8491D', hatch="/", linewidth=2)
    plt.bar(NMI_x, height=NMI_y, label="NMI", width=barWidth, color='white', edgecolor='#EEA23B', linewidth=2)
    plt.bar(ACC_x, height=ACC_y, label="ACC", width=barWidth, color='white', edgecolor='#4A7D90', hatch="//", linewidth=2)

    cout=0
    for x1, yy in zip(ARI_x, ARI_y):
        cout+=1
        if cout==2:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)
    cout=0
    for x1, yy in zip(NMI_x, NMI_y):
        cout+=1
        if cout==2:
            plt.text(x1, yy,"{:.3f}".format(yy), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)
    cout=0
    for x1, yy in zip(ACC_x, ACC_y):
        cout+=1
        if cout==2:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)

    plt.xticks([x+barWidth/2*(labelNums-1) for x in range(seriesNums)], ["r=2","r=4","r=6","r=8"])
    plt.xlabel("低秩矩阵维度r",fontproperties=myfont, fontsize=12)
    plt.ylabel("聚类性能",fontproperties=myfont, fontsize=12)
    plt.legend(loc = (0, 0.8))
    plt.savefig("./draw_r_zhuzhuang.png", dpi=300)

def draw_alpha_zhuzhuang():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)

    totalWidth=0.65 
    labelNums=3 
    barWidth=totalWidth/labelNums 
    seriesNums=3 

    ARI_x=[x for x in range(seriesNums)]
    NMI_x=[x+barWidth for x in range(seriesNums)]
    ACC_x=[x+2*barWidth for x in range(seriesNums)]

    ARI_y=[0.4219, 0.4260, 0.4143]
    NMI_y=[0.4653, 0.4689, 0.4579]
    ACC_y=[0.6411, 0.6477, 0.6330]

    plt.bar(ARI_x, height=ARI_y, label="ARI", width=barWidth, color='white', edgecolor='#4A7D90', hatch="/", linewidth=2)
    plt.bar(NMI_x, height=NMI_y, label="NMI", width=barWidth, color='white', edgecolor='#939249', linewidth=2)
    plt.bar(ACC_x, height=ACC_y, label="ACC", width=barWidth, color='white', edgecolor='#C24F1A', hatch="//", linewidth=2)

    cout=0
    for x1, yy in zip(ARI_x, ARI_y):
        cout+=1
        if cout==2:
            plt.text(x1, yy, "{:.3f}".format(yy), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)
    cout=0
    for x1, yy in zip(NMI_x, NMI_y):
        cout+=1
        if cout==2:
            plt.text(x1, yy, "{:.3f}".format(yy), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)
    cout=0
    for x1, yy in zip(ACC_x, ACC_y):
        cout+=1
        if cout==2:
            plt.text(x1, yy, "{:.3f}".format(yy), ha='center', va='bottom', fontsize=8, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, round(yy,3), ha='center', va='bottom', fontsize=8, rotation=0)

    plt.xticks([x+barWidth/2*(labelNums-1) for x in range(seriesNums)], ["α=0.5","α=1","α=2"])
    plt.xlabel("全局知识纠偏项权重α",fontproperties=myfont, fontsize=12)
    plt.ylabel("聚类性能",fontproperties=myfont, fontsize=12)
    plt.legend()
    plt.savefig("./draw_alpha_zhuzhuang.png", dpi=300)

def draw_M_zhuzhuang():
    plt.switch_backend('agg')
    plt.clf()
    fname = "/home/chenyannan/anaconda3/envs/pytorch_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
    myfont = FontProperties(fname=fname)

    totalWidth=0.8
    labelNums=3 
    barWidth=totalWidth/labelNums 
    seriesNums=4 

    ARI_x=[x for x in range(seriesNums)]
    NMI_x=[x+barWidth for x in range(seriesNums)]
    ACC_x=[x+2*barWidth for x in range(seriesNums)]

    # ARI_y_noper=[0.2862, 0.2426, 0.2421, 0.2168]
    # NMI_y_noper=[0.3985, 0.3343, 0.3494, 0.3268]
    # ACC_y_noper=[0.4906, 0.4445, 0.4401, 0.4413]

    # ARI_y=[0.5669, 0.5178, 0.4374, 0.4247]
    # NMI_y=[0.6159, 0.5601, 0.4984, 0.4859]
    # ACC_y=[0.7381, 0.7204, 0.6405, 0.6393]
    ARI_y=[0.5230, 0.4260, 0.4030, 0.3733]
    NMI_y=[0.5580, 0.4689, 0.4548,  0.4298 ]
    ACC_y=[0.7249, 0.6477,  0.6187, 0.5952]

    # plt.bar(ARI_x, ARI_y,  label="ARI", width=barWidth, edgecolor='#ffb6c1', color="white", hatch='/')
    # plt.bar(NMI_x, NMI_y,  label="NMI", width=barWidth, edgecolor='gold', color="white", hatch='/')
    # plt.bar(ACC_x, ACC_y,  label="ACC", width=barWidth, edgecolor='lightskyblue', color="white", hatch='/')

    # plt.bar(ARI_x, ARI_y_noper, label="ARI", width=barWidth, color='#ffb6c1')
    # plt.bar(NMI_x, NMI_y_noper, label="NMI", width=barWidth, color='gold')
    # plt.bar(ACC_x, ACC_y_noper, label="ACC", width=barWidth, color='lightskyblue')

    plt.bar(ARI_x, ARI_y,  label="ARI", width=barWidth, color='white', edgecolor='#DC0502', hatch="/", linewidth=2)
    plt.bar(NMI_x, NMI_y,  label="NMI", width=barWidth, color='white', edgecolor='#4174B3', linewidth=2)
    plt.bar(ACC_x, ACC_y,  label="ACC", width=barWidth,  color='white', edgecolor='#5A4D88', hatch="//", linewidth=2)

    cout=0
    for x1, yy in zip(ARI_x, ARI_y):
        cout+=1
        if cout==1:
            plt.text(x1, yy, "{:.3f}".format(yy), ha='center', va='bottom', fontsize=7.5, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy,  "{:.3f}".format(yy), ha='center', va='bottom', fontsize=8, rotation=0)
    cout=0
    for x1, yy in zip(NMI_x, NMI_y):
        cout+=1
        if cout==1:
            plt.text(x1, yy, "{:.3f}".format(yy), ha='center', va='bottom', fontsize=7.5, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy, "{:.3f}".format(yy), ha='center', va='bottom', fontsize=8, rotation=0)
    cout=0
    for x1, yy in zip(ACC_x, ACC_y):
        cout+=1
        if cout==1:
            plt.text(x1, yy, "{:.3f}".format(yy), ha='center', va='bottom', fontsize=7.5, rotation=0, fontweight='heavy')
        else:
            plt.text(x1, yy,  "{:.3f}".format(yy), ha='center', va='bottom', fontsize=8, rotation=0)

    plt.xticks([x+barWidth/2*(labelNums-1) for x in range(seriesNums)], ["M=3","M=4","M=5","M=6"])
    plt.xlabel("客户端拥有的数据类别数M",fontproperties=myfont, fontsize=12)
    plt.ylabel("聚类性能",fontproperties=myfont, fontsize=12)
    plt.legend(loc = (0.955, 0.9))
    plt.savefig("./draw_M_zhuzhuang.png", dpi=300)


if __name__ == '__main__':
    # drawNoclusterVScluster()
    # drawEpsilon()
    # drawR()
    # drawM()
    # drawShoulian()
    # drawAlpha()
    # drawShoulian100()
    # draw_epsilon_zhuzhuang()
    # draw_r_zhuzhuang()
    draw_alpha_zhuzhuang()
    draw_M_zhuzhuang()
    # from scipy.interpolate import lagrange

    # x = [1, 2, 3]
    # y = [9/16, 1/4, 1/16]
    # f = lagrange(x, y)
    # print(f(0))