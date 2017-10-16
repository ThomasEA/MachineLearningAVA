# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:13:44 2017

@author: Everton
"""
import numpy as np

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

array = randrange(40, 0, 1)
