import matplotlib.pyplot as plt
import numpy as np

# #matplotlib
# # x=np.linspace(0,5,10)
# # y=x**2
# # plt.plot(x,y)
# # plt.show()
#
#
# x=[1,2,3,4,5]
# x1=[1,2,3,4,5]
# y=[num**2 for num in x]
# z=[num*4.23 for num in x]
#
#
# # fig,axes=plt.subplots(1,2)
# # axes[0].plot(x,y)
# # axes[1].plot(y,x)
# # plt.show()
#
#
# fig,axes=plt.subplots(1,2)
# axes[0].plot(x,x1,label='x and x1')
# axes[0].plot(x,y,label='c squared')
# axes[1].plot(x,x1,label='x and x1')
# axes[1].plot(x,z,label='x and z')
# axes[0].set_title('First')
# axes[0].legend(loc=0)
# axes[1].legend(loc=0)
# plt.show()
#
#
# # fig=plt.figure()
# # axes=fig.add_axes([0,0,1,1])
# # axes.plot(y,x)
# # axes.set_title('Title')
# # axes.set_ylabel('Y')
# # axes.set_xlabel('xlabel', fontsize=10)
# # plt.show()


# x=[1,2,3]
# y=[num**2 for num in x]
# z=[num**(1/8) for num in y]
#
# fig,axes=plt.subplots(1,2)
# axes[0].plot(x,y,label='X and Y')
# axes[0].plot(z,x,label='Z and X')
# axes[0].set_title('X and Y')
# axes[0].set_xlabel('X')
# axes[0].set_ylabel('Y')
# axes[1].plot(x,z,label='X and Z')
# axes[1].plot(z,y,label='Z and Y')
# axes[1].set_title('X and Y and Z')
# axes[0].legend(loc=0)
# axes[1].legend(loc=0)
# plt.show()


x=[1,2,3,4,5]
y=[num**2 for num in x]

fig,axes=plt.subplots(1,2)
axes[0].plot(x)
axes[0].set_title('X')
axes[1].plot(y,label='Y')
axes[1].plot(y,x,label='Y and X')
axes[1].set_title('Y and Y,X')
axes[1].legend(loc=0)
plt.show()
