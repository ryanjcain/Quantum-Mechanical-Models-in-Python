'''
A Numerical Analysis of the Time Dependent Schrodinger Eauqion

Computer Science Final Project
Ryan Cain
April 21, 2016
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft, ifft #Fast Fourier and Inverse Fourier Algorithims!

class Schrodinger(object):
    def __init__(self, x, psi_x0, V_x, k0=None,
                hbar=1, m=1, t0=0.0):

        """
        ____________________________________________________________
        Parameters:
        ____________________________________________________________
        x : a numpy array (length-N) of evenly spaced floats
            (x spacing on matplot graph)
        psi_x0 : a numpy array of complex numbers
            (an array of values of the initial wave function at t0)
        V_x : a numpy array of floats
            (array of length-N for potential at each x coordinate)
        k0 : float
            (minimum vallue of k)
            k0 < k < 2*pi / dx
        hbar : float
            (planck's constant divided by 2pi defaulted here to 1)
        m : float
            (particle mass [default = 1])
        t0 : float
            (time 0 = 0)
        ____________________________________________________________
        """

        #Defining x, psi_x0, and V_x as numpy arrays:
        #--------------------------------------------------------------------#
        #   map([function], [iterable]): applies a function to each iterable
        #   numpy.asarray(a): creates an "array interpretation" of 'a'
        #   numpy.array.size: returns the number of elements in an array
        #--------------------------------------------------------------------#
        self.x, psi_x0, self.V_x = map(np.asarray, (x, psi_x0, V_x))
        N = self.x.size

        #Validate the array inputs:
        #--------------------------------------------------------------------#
        #   "assert" breaks the code if the folloiwng lines are not True
        #   numpy.array.shape: returns tuple of array dimensions (rows,columns)
        #--------------------------------------------------------------------#
        assert self.x.shape == (N,)
        assert psi_x0.shape == (N,)
        assert self.V_x.shape == (N,)


        #Iinitialize parameters:
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.dt_ = None #Check this syntax
        self.N = len(x)
        self.dx = self.x[1] - self.x[0] #difference between points in numpy array
        self.dk = 2 * np.pi / (self.N * self.dx) #full wavelength / total x space distance



        #Set initial momentum scale
        #numpy.arrange is range() but returns ndarray instead of list
        if k0 == None:
            self.k0 = -0.5 * self.N * self.dk #total k space length
        else:
            self.k0 = k0
        #Create an array of ALL the k values to be used
        self.k = self.k0 + self.dk * np.arange(self.N)
        self.psi_x = psi_x0


        self.compute_k_from_x()
        #Variables which hold steps in evolution of wave function???
        self.x_evolve_half = None
        self.x_evolve = None
        self.k_evolve = None


        #Attributes for dynamic plotting
        #--------------------------------------------------------------------#
        #   These are later assigned to a Line class so matplotlib methods
        #   Line.setdata() and line.plot() can be used
        #--------------------------------------------------------------------#
        self.psi_x_line = None
        self.psi_k_line = None
        self.V_x_line = None


    ##### METHODS #####

    #Create the integrand for the forward transform (x -> k)
    def set_psi_x(self, psi_x):
        self.psi_mod_x = (psi_x * np.exp(-1j * self.k[0] * self.x)
                            * self.dx / np.sqrt(2 * np.pi))

    #Inverse to set_psi_x : gets an updated psi_x value by canceling everything
    #else in the integrand
    def get_psi_x(self):
        return (self.psi_mod_x * np.exp(1j * self.k[0] * self.x)
                * np.sqrt(2 * np.pi) / self.dx)

    #Create the integrand for the inverse transform (k -> x)
    def set_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * self.x[0]
                                            * self.dk * np.arange(self.N))

    #Inverse to set_psi_k : gets updated psi_k value?
    def get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * self.x[0]
                                            * self.dk * np.arange(self.N))

    def _get_dt(self):
        return self.dt_ #This value of dt is local to the class


    def _set_dt(self, dt):
        """
        Computes attributes to be used in time_step():
            half step in x
            full step in x
            full tep in x


        Parameter: dt (time step)
        Returns: None
        """

        if dt != self.dt_:
            self.dt_ = dt
            #Time step solution to V part of x-space Schrodinger (worked out in notes)
            self.x_evolve_half = np.exp(-0.5 * 1j * self.V_x
                                        / self.hbar * dt)
            self.x_evolve = self.x_evolve_half * self.x_evolve_half
            #Time step solution to other part of schrodinger in K space
            self.k_evolve = np.exp(0.5 * -1j * self.hbar /
                                    self.m * (self.k * self.k) * dt)

    #Anytime psi_x is set to something, these functions are called
    psi_x = property(get_psi_x, set_psi_x)
    psi_k = property(get_psi_k, set_psi_k)
    dt = property(_get_dt, _set_dt)


    #Fast Fourier Transforms!!!!
    def compute_k_from_x(self):
        self.psi_mod_k = fft(self.psi_mod_x)

    def compute_x_from_k(self):
        self.psi_mod_x = ifft(self.psi_mod_k)

    def time_step(self, dt, Nsteps=1):
        """
        Perform a series of time-steps with leapfrong integration based on the
        solutions to the time-dependent Scrodinger Equation.
        (worked out in notebook)

        These values all depend on dt.  Calculating them up front, while dt
        updates is more efficient.

        Parameters:
        __________

        dt : float
            small time interval to integrate
        Nsteps : float, optional
            the number of intervals to compute
        """

        self.dt = dt

        if Nsteps > 0:
            self.psi_mod_x *= self.x_evolve_half

        for i in xrange(Nsteps - 1):
            self.compute_k_from_x()
            self.psi_mod_k *= self.k_evolve
            self.compute_x_from_k()
            self.psi_mod_x *= self.x_evolve

        self.compute_k_from_x()
        self.psi_mod_k *= self.k_evolve

        self.compute_x_from_k()
        self.psi_mod_x *= self.x_evolve_half

        self.compute_k_from_x()

        self.t += dt * Nsteps


###############################################################################
#   Guassian Waves:                                                           #
###############################################################################

#Inital Wave forms forms!

def gauss_x(x, a, x0, k0):
    """
    A gaussian wave packet of width a, centered at x0, with momentum k0

    Note: This is a guassian wave packet with typical guassian component
    and a cosine wave!  (From Dr. Kellogg)
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

def gauss_k(k, a, x0, k0):
    """
    Fourier transfomr of guass_x(x)
    """

    return ((a / np.sqrt(np.pi)) ** 0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))




###############################################################################
#   Animation Functions:                                                      #
###############################################################################

def theta(x):
    """
    returns 0 if x <= 0, and 1 if x > 0
    """
    #Array math is better:
    x = np.asarray(x)
    y = np.zeros(x.shape)
    #Set y equal to 1 if x > 0!!!! SLICK!!!!
    y[x > 0] = 1.0
    return y

def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))

def parabolic_well(arg):
    pass
###############################################################################
# Create the Animation                                                        #
###############################################################################

# Specify time steps and duration
dt = 0.01
N_steps = 50
t_max = 120
frames = int(t_max / float(N_steps * dt))

# Constants
hbar = 1.0
m = 1.9

# Range of x coordinates
N = 2 ** 11
dx = 0.1
x = dx * (np.arange(N) - 0.5 * N) #Create array of all x's

# specify potential
V0 = 1.5
L = hbar / np.sqrt(2 * m * V0)
a = 3 * L
x0 = -60 * L
V_x = square_barrier(x, a, V0)
V_x[x < -98] = 1E6
V_x[x > 98] = 1E6

# Initial momentum and derived quantities
p0 = np.sqrt(2 * m * 0.2 * V0)
dp2 = p0 * p0 * 1./80
d = hbar / np.sqrt(2 * dp2)

k0 = p0 / hbar
v0 = p0 / m

#Create the inital x-space wave packet!!
psi_x0 = gauss_x(x, d, x0, k0)


# Create a Schrodinger object to perform calculations!!!

S = Schrodinger(x=x,
                psi_x0=psi_x0,
                V_x=V_x,
                hbar=hbar,
                m=m,
                k0=-28)


###############################################################################
#   Matplotlib Animation:                                                     #
###############################################################################

fig = plt.figure()

# plotting limits
xlim = (-100, 100)
klim = (-5, 5)

# top axes show x-space data #AND OMG LATEX IN MATPLOTLIB!!!!
ymin = 0
ymax = V0
ax1 = fig.add_subplot(211, xlim=xlim,
                      ylim=(ymin - 0.2 * (ymax - ymin),
                            ymax + 0.2 * (ymax - ymin)))
psi_x_line, = ax1.plot([], [], c='r', label=r'$|\psi(x)^2|$')
V_x_line, = ax1.plot([], [], c='k', label=r'$V(x)$')
center_line = ax1.axvline(0, c='k', ls=':',
                          label = r"$x_0 + v_0t$")

title = ax1.set_title("")
ax1.legend(prop=dict(size=12))
ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$|\psi(x)|$')

# bottom axes show k-space
ymin = abs(S.psi_k).min()
ymax = abs(S.psi_k).max()
ax2 = fig.add_subplot(212, xlim=klim,
                      ylim=(ymin - 0.2 * (ymax - ymin),
                            ymax + 0.2 * (ymax - ymin)))
psi_k_line, = ax2.plot([], [], c='r', label=r'$|\psi(k)|$')

p0_line1 = ax2.axvline(-p0 / hbar, c='k', ls=':', label='r$\pm p_0$')
p0_line2 = ax2.axvline(p0 / hbar, c='k', ls=':')
mV_line = ax2.axvline(np.sqrt(2 * V0) / hbar, c='k', ls='--',
                      label=r'$\sqrt{2mV_0}$')
ax2.legend(prop=dict(size=12))
ax2.set_xlabel('$k$')
ax2.set_ylabel(r'$|\psi(k)|$')

V_x_line.set_data(S.x, S.V_x)


###############################################################################
#   ANIMATE IT!!                                                              #
###############################################################################
def init():
    psi_x_line.set_data([], [])
    V_x_line.set_data([], [])
    center_line.set_data([], [])

    psi_k_line.set_data([], [])
    title.set_text("")
    return (psi_x_line, V_x_line, center_line, psi_k_line, title)


def animate(i):
    S.time_step(dt, N_steps)
    psi_x_line.set_data(S.x, 4 * abs(S.psi_x))
    V_x_line.set_data(S.x, S.V_x)
    center_line.set_data(2 * [x0 + S.t * p0 / m], [0, 1])

    psi_k_line.set_data(S.k, abs(S.psi_k))
    title.set_text("t = %.2f" % S.t)
    return (psi_x_line, V_x_line, center_line, psi_k_line, title)


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=False)


# save the animation as an mp4.
#anim.save('Schrodinger.mp4', fps=30)


#Run the matplotlib program
plt.show()
