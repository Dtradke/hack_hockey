'''
    File name: plot_2d_beta_figures.py
    Author: Daniel Radke
    Date created: 01.05.2020
    Date last modified: 01.05.2020
    Python Version: 3.7.3
'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import copy
# import beta_functions2 as bf2

def angle_between_vectors(a,b):
    return np.arccos(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))

def normalize(vec):
    return vec / np.linalg.norm(vec)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.grid()
# ax.set_xlabel('x')
# ax.set_ylabel('y')
ax.axis('equal')
ax.axis('off')
# lims = 1.45
# ax.set_ylim(-lims,lims)
# ax.set_xlim(-lims,lims)

# straight lines
y = np.arange(-10, 10, 1)
x1 = np.repeat(-1,len(y))
x2 = np.repeat(1,len(y))



# ax.plot(x1, y,'--',linewidth=1,c='0.3')
# ax.plot(x2, y,'--',linewidth=1,c='0.3')
# ax.fill_betweenx(y, x1, x2, facecolor='g', alpha=0.1)


# beta
# {

# beta = 1/2
u = np.linspace(np.pi/2,5*np.pi/2,3600)

x = 2*np.sin(u)

y1 = 2*np.cos(u) - np.sqrt(3) #flip these, might be beta=2?
y2 = 2*np.cos(u) + np.sqrt(3)

x[x>0] = 0 # make half circles


# ax.plot(x[y1<=0], y1[y1<=0],'--',linewidth=1,c='0.3')
# ax.plot(x[y2>=0], y2[y2>=0],'--',linewidth=1,c='0.3')
ax.plot(x, y1,'--',linewidth=1,c='0.3')
ax.plot(x, y2,'--',linewidth=1,c='0.3')
l1 = len(y1[y1<=0])
l2 = len(y2[y2>=0])
ax.fill_between(x[y1<=0],y1[y1<=0],np.repeat(0,l1), facecolor='b', alpha=0.2)
ax.fill_between(x[y2>=0],y2[y2>=0],np.repeat(0,l2), facecolor='b', alpha=0.2)


x_pt = 0
for count, i in enumerate(y2):
    if i <= 0 and x[count] < x_pt:
        x_pt = x[count]




# print(np.where(y1==0))

# print(y1[np.argmin(np.absolute(y1))])
# print(x[np.argmin(np.absolute(y1))])
# exit()

# beta = 1
# x = np.sin(u)
# y = np.cos(u)
#
# ax.plot(x,y,'--',linewidth=1,c='0.3')
# c = patches.Circle((0,0),radius=1,facecolor='b',alpha=0.2)
# ax.add_patch(c)
#
# # beta = 2
# u = np.linspace(0,2*np.pi,3600)
# b = 2
# x1 = b*np.sin(u) + (b-1)
# x2 = b*np.sin(u) - (b-1)
# y = b*np.cos(u)
#
# ax.plot(x1[x1<=0],y[x1<=0],'--',linewidth=1,color = '0.3')
# ax.plot(x2[x2>=0],y[x2>=0],'--',linewidth=1,color = '0.3')
# l1 = len(x1[x1<=0])
# l2 = len(x2[x2>=0])
# ax.fill_betweenx(y[x1<=0],x1[x1<=0],np.repeat(0,l1), facecolor='b', alpha=0.2)
# ax.fill_betweenx(y[x2>=0],x2[x2>=0],np.repeat(0,l2), facecolor='b', alpha=0.2)
#
# # beta = 3
# u = np.linspace(0,2*np.pi,3600)
# b = 3
# x1 = b*np.sin(u) + (b-1)
# x2 = b*np.sin(u) - (b-1)
# y = b*np.cos(u)
#
# ax.plot(x1[x1<=0],y[x1<=0],'--',linewidth=1,color = '0.3')
# ax.plot(x2[x2>=0],y[x2>=0],'--',linewidth=1,color = '0.3')
# l1 = len(x1[x1<=0])
# l2 = len(x2[x2>=0])
# ax.fill_betweenx(y[x1<=0],x1[x1<=0],np.repeat(0,l1), facecolor='b', alpha=0.2)
# ax.fill_betweenx(y[x2>=0],x2[x2>=0],np.repeat(0,l2), facecolor='b', alpha=0.2)

ax.scatter([x_pt,0],[0,0],color='k',marker='o',edgecolors='k',zorder=10, s=100)
ax.annotate("p",(x_pt-0.55, 0), fontsize=20)
# ax.annotate("q",(1.35, 0), fontsize=20)
ax.annotate("r",(0.25, 0), fontsize=20)

# plt.show()
fname = 'betas/beta_union3.png'
plt.savefig(fname,bbox_inches='tight', dpi=300)
plt.close()
# '''
# }


# EGG alpha
# {
exit()
# alpha = 1/2
el = patches.Ellipse((0,0),width=2,height=1,facecolor='none',edgecolor='0.3',linewidth=1,linestyle='--')
ax.add_patch(el)
e = patches.Ellipse((0,0),width=2,height=1,facecolor='r',alpha=0.2)
ax.add_patch(e)

# alpha = 1
el = patches.Ellipse((0,0),width=2,height=2,facecolor='none',edgecolor='0.3',linewidth=1,linestyle='--')
ax.add_patch(el)
e = patches.Ellipse((0,0),width=2,height=2,facecolor='r',alpha=0.2)
ax.add_patch(e)

# alpha = 2
el = patches.Ellipse((0,0),width=2,height=4,facecolor='none',edgecolor='0.3',linewidth=1,linestyle='--')
ax.add_patch(el)
e = patches.Ellipse((0,0),width=2,height=4,facecolor='r',alpha=0.2)
ax.add_patch(e)

# alpha = 3
el = patches.Ellipse((0,0),width=2,height=6,facecolor='none',edgecolor='0.3',linewidth=1,linestyle='--')
ax.add_patch(el)
e = patches.Ellipse((0,0),width=2,height=6,facecolor='r',alpha=0.2)
ax.add_patch(e)

ax.scatter([-1,1],[0,0],color='w',marker='o',edgecolors='k',zorder=10)

'''
# }


# theta phi Ellipse
# {
'''
# axis
axisLen = 1
ax.plot([-axisLen-0.25,axisLen+0.25], [0,0],'-',linewidth=1,c='0.4')
ax.plot([0,0], [-axisLen,axisLen],'-',linewidth=1,c='0.4')

# ellipse
el = patches.Ellipse((0,0),width=2,height=1.5,facecolor='none',edgecolor='k',linewidth=1.5,linestyle='-')
ax.add_patch(el)

# points
x = 0.35
y = np.sqrt( (1- (x)**2 ) * 0.75**2 )
ax.scatter([-1,1,x],[0,0,y],color='w',marker='o',edgecolors='k',zorder=10)
# ax.scatter([x],[y],color=(1,1,1),marker='o',edgecolors='k')

theta = bf2.angle_between_vectors(np.array([x,y])-np.array([-1,0]),np.array([1,0]))
phi = bf2.angle_between_vectors(np.array([x,y])-np.array([1,0]),np.array([-1,0]))

# lines
ax.plot([-1,x,1],[0,y,0],'-k',linewidth=1)
ax.plot([x,x],[y,0],'--k',linewidth=1)

# angles
u = np.linspace(0,theta,20)
ax.plot(0.3*np.cos(u)-1,0.3*np.sin(u),'-k')
u = np.linspace(np.pi-phi,np.pi,20)
ax.plot(0.3*np.cos(u)+1,0.3*np.sin(u),'-k')


'''
# }


# denoise
# {
# '''
def rotate2(pts,delta):
    delta = np.radians(delta)
    rMat = np.asarray([[np.cos(delta),-np.sin(delta)],[np.sin(delta),np.cos(delta)]])
    return (np.matmul(pts,rMat))

# left
y = np.linspace(-1.5,1.5,101)
a = 1
b = 10
x = (1 + (y/b)**2) * (a**2)
ax.plot(-x,y,'-k')
noise = np.array([0,0,0,0,0,0,0.13,0,0,0,0])
LsampleY = np.linspace(-1.5,1.5,11)
LsampleX = (1 + (LsampleY/b)**2) * (a**2)
ax.scatter(-LsampleX + noise,LsampleY,color='w',marker='o',edgecolors='k',zorder=10)



# right
y = np.linspace(-1.5,1.5,101)
a = 1
b = 2.5
x = (1 + (y/b)**2) * (a**2)
xy = np.column_stack((x,y-0.4))
xy = rotate2(xy,20)
ax.plot(xy[:,0],xy[:,1],'-k')

RsampleY = np.linspace(-1.5,1.5,11)
RsampleX = (1 + (RsampleY/b)**2) * (a**2)
RsampleY = RsampleY
RsampleXY = np.column_stack((RsampleX,RsampleY-0.4))
RsampleXY = rotate2(RsampleXY,20)
# ax.scatter(RsampleX,RsampleY,color='w',marker='o',edgecolors='k',zorder=10)
ax.scatter(RsampleXY[:,0],RsampleXY[:,1],color='w',marker='o',edgecolors='k',zorder=10)



# sampleY = np.linspace(-1.5,1.5,11)
# sampleX = (1 + (sampleY/b)**2) * (a**2)
# sampleX1 = sampleX + np.random.randn(len(sampleX)) * 0.03
# sampleX2 = -sampleX + np.random.randn(len(sampleX)) * 0.03
# noise = np.array([0,0,0,0,0,0,0.13,0,0,0,0])
# ax.scatter(sampleX,sampleY-0.2,color='w',marker='o',edgecolors='k',zorder=10)
# ax.scatter(sampleX - 2 + noise,sampleY,color='w',marker='o',edgecolors='k',zorder=10)


p = np.array([-LsampleX[5],LsampleY[5]])
q1 = np.array(RsampleXY[6])
q2 = np.array([-LsampleX[6]+0.13,LsampleY[6]])
ax.scatter(p[0],p[1],color='w',marker='o',linewidths=2,edgecolor='r',zorder=11)
ax.scatter(q1[0],q1[1],color='w',marker='o',linewidths=2,edgecolor='g',zorder=11)
ax.scatter(q2[0],q2[1],color='w',marker='o',linewidths=2,edgecolor='g',zorder=11)
ax.plot([p[0],p[0]-1],[p[1],p[1]],'-',c='r',linewidth=3,zorder=9)
ax.plot([-1.5,1.5],[0,0],'--k',linewidth=0.7)


# big circle
t1 = RsampleXY[6]
slope = -(p[0]-t1[0]) / (p[1]-t1[1])
midp = np.mean((p,t1),axis=0)
# y - midp[1] = slope(x - midp[0])
x1 = (-midp[1] + slope*midp[0]) / slope

ax.scatter(x1,[0],color='w',marker='o',linewidths=2,edgecolor='b',zorder=11)
c = patches.Circle((x1,0),radius=1+x1,facecolor='k',alpha=0.2)
ax.add_patch(c)



# small circle
t2 = np.array([(-LsampleX + noise)[6],LsampleY[6]])

slope = -(p[0]-t2[0]) / (p[1]-t2[1])
midp = np.mean((p,t2),axis=0)
# y - midp[1] = slope(x - midp[0])
x2 = (-midp[1] + slope*midp[0]) / slope

ax.scatter(x2,[0],color='w',marker='o',linewidths=2,edgecolor='b',zorder=11)
c = patches.Circle((x2,0),radius=1+x2,facecolor='k',alpha=0.2)
ax.add_patch(c)
# '''
# }



# lines and angles
ax.plot([x1,q1[0]],[0,q1[1]],':k')
ax.plot([x2,q2[0]],[0,q2[1]],':k')

theta1 = angle_between_vectors(q1-[x1,0],[1,0])
theta2 = angle_between_vectors(q2-[x2,0],[-1,0])

u = np.linspace(np.pi, 2*np.pi-theta1, 30)
ax.plot(0.1*np.cos(u),0.1*np.sin(u),'-k')
u = np.linspace(np.pi-theta2,np.pi,30)
ax.plot(0.1*np.cos(u)+x2,0.1*np.sin(u),'-k')

plt.show()
