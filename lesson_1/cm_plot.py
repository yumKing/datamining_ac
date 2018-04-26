def cm_plot(y, yp):
    # 导入混淆矩阵函数
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yp)

    # 导入作图库
    import matplotlib.pyplot as plt
    # 画混淆矩阵图
    plt.matshow(cm, cmap=plt.get_cmap('Greens'), fignum=1)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    return plt

