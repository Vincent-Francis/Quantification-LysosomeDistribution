import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy

colors=['k','r','g','b','m','y','c']
variants=[]
path="./output/"
for count,folder in enumerate(os.listdir(path)):
    print(folder)
    v=folder.strip().split("-")
    variants.append(v[6]+"-"+v[7])
    loaded=None
    for file in os.listdir(os.path.join(path,folder)):
        if "simplified_matrix_summary_normalized.npy" in file:
            loaded=np.load(os.path.join(path,folder,file))
    difference=loaded-np.roll(loaded,1,axis=1)
    difference[:,0]=loaded[:,0]
    for item in range(loaded.shape[0]):
        plt.figure(1)
        plt.scatter(range(loaded.shape[1]),loaded[item,:],color=colors[count])
        plt.title("Data points scatter")
    plt.figure(2)
    plt.scatter(np.dot(5,range(1,loaded.shape[1]+1)),np.mean(loaded,axis=0),color=colors[count],label=variants[count])
    plt.plot(np.dot(5,range(1,loaded.shape[1]+1)),np.mean(loaded,axis=0),color=colors[count])
    plt.scatter(np.dot(5,range(1,loaded.shape[1]+1)),(np.std(loaded,axis=0)/np.sqrt(loaded.shape[0]))+np.mean(loaded,axis=0),color=colors[count],marker="_",linewidths =20,s=50)
    plt.scatter(np.dot(5,range(1,loaded.shape[1]+1)),-1*(np.std(loaded,axis=0)/np.sqrt(loaded.shape[0]))+np.mean(loaded,axis=0),color=colors[count],marker="_",linewidths =20,s=50)
    plt.ylabel('Cumulative lamp intensity')
    plt.xlabel('Distance from center to periphery (%)')
    plt.title("Lysosome distribution")
    plt.legend(bbox_to_anchor=(1.02,1), loc=2, mode="expand", borderaxespad=0.)
    plt.savefig("Lysosome-cumulative-lamp-intensity.png")

    plt.figure(3)
    plt.scatter(range(loaded.shape[1]),np.mean(difference,axis=0),color=colors[count],label=variants[count])
    plt.plot(range(loaded.shape[1]),np.mean(difference,axis=0),color=colors[count])
    plt.title("Difference ")
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, mode="expand", borderaxespad=0.)

plt.show()
