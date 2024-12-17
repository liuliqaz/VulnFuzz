import matplotlib.pyplot as plt

if __name__ == "__main__":
    #test = [44,32,8,0]
    #test = [38,29,11,2]
    #test = [27,35,5,1]
    test = [328,319,31,23]
    TP = test[0]
    TN = test[1]
    FP = test[2]
    FN = test[3]


    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    print(acc, precision, recall, f1)
    tpr = (TP / (TP + FN))
    print("TPR:", str(tpr))
    # plt.rcParams['font.family'] = ['SimHei']
    #
    # x = [2015, 2016, 2017, 2018, 2019]  # x轴坐标数据
    # y = [500, 512, 514, 530, 516]  # y轴坐标数据
    # y1 = [520, 521, 526, 545, 563]
    # plt.plot(x, y, 'b', label='wenke', linewidth=2)  # 绘制线段
    # plt.plot(x, y1, 'm', label='like', linewidth=2)
    # plt.title('绘制折线图')  # 添加图表标题
    #
    # plt.ylabel('成绩')  # 添加y轴标题
    # plt.xlabel('年份')  # 添加x轴标题
    #
    # plt.legend()  # 设置图例
    # plt.legend()
    # plt.savefig('折线图', dip=72)  # 以分辨率72来保存图片
    #
    # plt.show()  # 显示图形

    # x = [i for i in range(11)]
    # a = [0,0.498116761,0.800376654,0.798964202,0.922787189,0.927495301,0.918079078,0.911016941,0.942090392,0.956214666,0.953860641]
    # b = [1,0.688539386,0.624838233,0.435452282,0.279030204,0.198983461,0.204102516,0.212847501,0.152129978,0.135594964,0.133403137]
    # plt.plot(x, a, label='acc', linewidth=2)
    # plt.plot(x, b, label='loss', linewidth=2)
    # plt.ylabel("acc/loss", fontsize=12)
    # plt.xlabel("epochs", fontsize=12)
    # plt.legend()
    # plt.show()
    # plt.savefig('折线图')