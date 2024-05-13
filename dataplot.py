import matplotlib.pyplot as plt  # Visualisering


# changes typ from dataframe to array
def dataplot(data, gest):
    gest_all = data[data['index'] == gest].to_numpy()

    print(len(gest_all))

    kropp = [11, 10, 1, 0]

    h_arm = [1, 3, 5, 7, 9]
    v_arm = [1, 2, 4, 6, 8]

    h_ben = [11, 12, 14, 16, 18]
    v_ben = [11, 13, 15, 17, 19]

    # empty figure to plot in
    fig = plt.figure(figsize=(12, 7))

    for j in range(0, len(gest_all)):  # plots all instances of gesture
        gest = gest_all[j]
        gest_f1 = gest[0:60]  # <-- choose frame

        x = gest_f1[0::3]
        y = gest_f1[1::3]
        z = gest_f1[2::3]
        # print(x,y,z, sep='\n') #prints coordinates

        ax = fig.add_subplot(5, 5, j+1, projection='3d')
        ax.scatter(x, y, z, s=2)
        ax.plot(x[kropp], y[kropp], z[kropp], label="Kropp")
        ax.plot(x[h_arm], y[h_arm], z[h_arm], label="Höger Arm")
        ax.plot(x[v_arm], y[v_arm], z[v_arm], label="Vänster Arm")
        ax.plot(x[h_ben], y[h_ben], z[h_ben], label="Höger Ben")
        ax.plot(x[v_ben], y[v_ben], z[v_ben], label="Vänster Ben")

        ax.view_init(elev=-70, azim=90)
        ax.legend(fontsize=2)

    plt.tight_layout()
    plt.show()
