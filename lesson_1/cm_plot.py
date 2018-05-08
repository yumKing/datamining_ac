def cm_plot(y, yp):
    # 导入混淆矩阵函数
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yp)

    # 导入作图库
    import matplotlib.pyplot as plt
    # 画混淆矩阵图
    # Create custom colormaps
    cdict = {'red': [(0.0, 0.0, 0.0),   # Full red at the first stop
                     (0.5, 1.0, 1.0),   # No red at second stop
                     (1.0, 1.0, 1.0)],  # Full red at final stop
             'green': [(0.0, 0.0, 0.0),  # No green at all stop
                       (0.5, 0.0, 0.0),
                       (1.0, 0.0, 0.0)],
             'blue': [(0.0, 0.0, 0.0),   # No blue at first stop
                      (0.5, 0.0, 0.0),   # Full blue at second stop
                      (1.0, 1.0, 1.0)]}  # No blue at final stop
    colorlist = ['#5fd9cd','#eaf786','#ffb5a1','#b8ffb8','#b8f4ff']
    from matplotlib.colors import LinearSegmentedColormap
    # cmap = LinearSegmentedColormap('Rd_Bl_Gr', cdict, 256)
    cmap = LinearSegmentedColormap.from_list('mylist',colorlist,gamma=2)
    # plt.matshow(cm, cmap=plt.get_cmap('Greens'), fignum=1)
    plt.matshow(cm, cmap=cmap, fignum=1)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(
                x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    return plt
