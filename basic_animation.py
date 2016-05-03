import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

#Create a Figure window with axes and instantiate a Line Object
fig = plt.figure()
ax = plt.axes(xlim=(0,2), ylim=(-2,2))
line, = ax.plot([], [], lw=2) #lw = linewidth. plot() is a Line Class



#Create a baseframe...make a background to plot.  Runs once at the beginning
#... accomplishes everything that needs to be done just once
def init():
    line.set_data([],[])
    return line,
    #return the line object so the animator knows which objects
    #on the plot to update after each frame

#Animation Function
def animate(i): #i is the frame number
    x = np.linspace(0, 2, 1000) #creates 1000 evently spaced ticks from [0,2]
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x,y)
    return line,

#call the animator class. blit=True means only re-draw the parts that
#have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=100, interval=20, blit=False)
#Dr. Kellogg says on mac... blit has to be set to False

# save the animation as an mp4.
anim.save('basic_animation.mp4', fps=30)
#Run the matplotlib program
plt.show()
