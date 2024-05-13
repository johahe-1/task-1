import Task_2_back

######################### plotting of 3d skeleton #########################

# changes typ from dataframe to array
gest_all = train_head[train_head['index'] == 5].to_numpy() #<-- ändra index för olika

print(len(gest_all))

Kropp = [11, 10, 1, 0]

H_arm = [1, 3, 5, 7, 9]
V_arm = [1, 2, 4, 6, 8]

H_ben = [11, 12, 14, 16, 18]
V_ben = [11, 13, 15, 17, 19]

# empty figure to plot in
fig = plt.figure(figsize=(12, 7))

for j in range(0,len(gest_all)): # plots all instances of gesture
    gest = gest_all[j]
    gest_f1 = gest[0:60]  #<-- choose frame

    x = gest_f1[0::3]
    y = gest_f1[1::3]
    z = gest_f1[2::3]
    #print(x,y,z, sep='\n') #prints coordinates

    ax = fig.add_subplot(5, 5, j+1, projection='3d')
    ax.scatter(x, y, z, s = 2)
    ax.plot(x[Kropp], y[Kropp], z[Kropp], label = "Kropp")
    ax.plot(x[H_arm], y[H_arm], z[H_arm], label = "Höger Arm")
    ax.plot(x[V_arm], y[V_arm], z[V_arm], label = "Vänster Arm")
    ax.plot(x[H_ben], y[H_ben], z[H_ben], label = "Höger Ben")
    ax.plot(x[V_ben], y[V_ben], z[V_ben], label = "Vänster Ben")

    ax.view_init(elev=-70, azim=90)
    ax.legend(fontsize=2)

plt.tight_layout()
plt.show()