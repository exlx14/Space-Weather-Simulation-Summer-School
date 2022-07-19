# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:21:35 2022

@author: Elle Lavichant 
"""
from math import factorial
from math import pi
import argparse
import numpy as np

def cos_approx(x, npts=10):
    """ This is the Taylor Series for the Cosine function """
    return sum([((-1)**n)/ (factorial((2*n))) *(x**(2*n)) for n in range(npts)])

# Will only run if this is run from command line as opposed to importe

def parse_args():



    # Create an argument parser:
    parser = argparse.ArgumentParser(description = \
                                     'This is a function for running the cos estimation problem ')
    
    
    # x: the angle to approximate:
    parser.add_argument('x', \
                        help = 'angle approximation', \
                        type=float, default= 0)
    # npts: number of points, type integer, default 10:
    parser.add_argument('npts', \
                        help = 'number of points (default = 10)', \
                        type = int, default = 10)
    


        
    # actually parse the data now:
    args = parser.parse_args()
    
    return args
    
if __name__ == '__main__':  # main code block
    args = parse_args()
    print(args)
    x=args.x
    print(x)
    npts = args.npts
    print(npts)
    print("cos(x) = ", cos_approx(x,npts))
    assert cos_approx(x,npts) < np.cos(x)+1.e-2 and cos_approx(x,npts)\
        > np.cos(x)-1.e-2, "This is not a good approximation"
    print("This is a good approximation")

    


    
    
