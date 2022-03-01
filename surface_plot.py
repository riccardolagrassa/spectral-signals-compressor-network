import sys
from matplotlib.pyplot import subplots, xticks, legend, show, set_cmap, axes

epsilon=sys.float_info.epsilon
import numpy as np
from matplotlib import cm
from mpl_toolkits import mplot3d



def plot_accuracy_epochs(X, Y, theta_list, Z_cosine):
    fig, ax = subplots()
    ax = axes(projection='3d')
    #ax.(domainX, domainY, theta_list, 50, cmap='binary')
    ax.plot_surface(X, Y, theta_list, cmap=cm.bwr,linewidth=0, antialiased=False)#label='gr='+str(i[2])
    ax.plot_surface(X, Y, Z_cosine, cmap=cm.bwr,linewidth=0, antialiased=False)#label='gr='+str(i[2])

    ax.set(xlabel='Domain', ylabel='Range', zlabel='error')
    #xticks(domain)
    #ax.axvline(x=std_median, ls='--', c='black', linewidth=2.0, label='std center')
    ax.grid()
    #legend(loc='lower right')
    #ax.axvline(inflection,c='black')
    #ax.axvline(0.0627, c='red')
    #fig.savefig('shift_curve.png')
    show(bbox_inches='tight', dpi=500)

from numpy import dot
from numpy.linalg import norm

theta_list=[]
domainX=np.arange(0, 255, 1)
domainY=np.arange(0, 255, 1)

#for k in [-0.07]:
# tmp=[]
# for i in domainX:
#     for j in domainY:
#         middle=[0.001, 0.001]
#         #abs(dot([i, j], middle)/(norm([i, j])*norm(middle)))
#         theta_list.append(np.subtract([i,j],middle).mean())

# for i in theta_list:
#     print(i)
# plot_accuracy_epochs(domain/(np.sqrt(((1 - 0) ** 2) / 4)),theta_list, median_std_value_norm)
X, Y = np.meshgrid(domainX,domainY)
middle = np.tile(np.ones(255)*0.001, (255, 1))
cosine=abs(np.dot(X, middle)/(norm(X)*norm(middle))) + abs(np.dot(Y, middle)/(norm(Y)*norm(middle)))
l1=abs(np.subtract(X,middle)) + abs(np.subtract(Y,middle))
Z= 0.1 * l1 #+ 5000* cosine
Z_cosine= 0.1 * l1 + 5000* cosine
plot_accuracy_epochs(X,Y,Z,Z_cosine)
