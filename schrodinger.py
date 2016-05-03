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

class Shrodinger(object):
    def __init__(self, x, psi_x0, V_x, k0=None,
                hbar=1, m=1, t0=0.0):

        """
        ------------------------------------------------------------
        Parameters:
        ------------------------------------------------------------
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
        ------------------------------------------------------------
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
        self.dt = None #Check this syntax
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
    '''def set_psi_k(self, psi_k):
        self.psi_mod_k = (psi_k * np.exp(1j * self.x[0]))'''
