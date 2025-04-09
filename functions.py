import numpy as np
rng = np.random.default_rng()
# Basic Function Definitions

def Q(deltaS, args):
    """
    Defines q in the system of ODES
    
    Parameters:
        deltaS: array, values to solve over
        args: array of form: [beta, alpha, deltaT, H, k] (takes H despite not using it for ease of using in other functions)
            all elements are floats
    """
    return args[4]*(args[1]*args[2] - args[0]*deltaS)


def dS1dt(deltaS, args):
    """
    Defines Equation 1 of the system of ODES
    
    Parameters:
        deltaS: array, values to solve over
        args: array of form: [beta, alpha, deltaT, H, k]
            all elements are floats
    """
    
    return -args[3] + np.absolute(Q(deltaS, args))*deltaS

def dS2dt(deltaS, args):
    """
    Defines Equation 2 of the system of ODES
    
    Parameters:
        deltaS: array, values to solve over
        args: array of form: [beta, alpha, deltaT, H, k]
            all elements are floats
    """
    
    return args[3] - np.absolute(Q(deltaS, args))*deltaS

def DeltaSdt(deltaS, args):
    """
    Defines the single equation ODE for Delta S rather than S1 and S2
    
    Parameters:
        deltaS: array, values to solve over
        args: array of form: [beta, alpha, deltaT, H, k]
            all elements are floats
    """
    return dS2dt(deltaS, args) - dS1dt(deltaS,args)

def AnalyticEQS(args):
    """
    Takes the given parameters and calculates deltaS for the 3 equilibrium points. Returns None for any imaginary solutions given the parameters.
    
    Parameters:
        args: array of form: [beta, alpha, deltaT, H, k]
            all elements are floats
    Output:
        Array: [eqP1, eqP2, eqP3], all values are a float if solution is real or None if it is imaginary
    """
    
    radicand1 = 1/4 - args[3]*args[0]/(args[4]*args[1]**2*args[2]**2)
    radicand2 = 1/4 + args[3]*args[0]/(args[4]*args[1]**2*args[2]**2)
    
    if radicand1 >= 0:
        eqP1 = args[1]*args[2] * (1/2 + np.sqrt(1/4 - args[3]*args[0]/(args[4]*(args[1]**2)*(args[2]**2)))) / args[0]
        eqP2 = args[1]*args[2] * (1/2 - np.sqrt(1/4 - args[3]*args[0]/(args[4]*(args[1]**2)*(args[2]**2)))) / args[0]
    else:
        eqP1 = None
        eqP2 = None
    
    if radicand2 >= 0:
        eqP3 = args[1]*args[2] * (1/2 + np.sqrt(1/4 + args[3]*args[0]/(args[4]*(args[1]**2)*(args[2]**2)))) / args[0]
    else:
        eqP3 = None
    
    return [eqP1, eqP2, eqP3]

def q_H(H,args):
    '''
    A Function of q in terms of H at equilibrium points:
        |q|dS = H => |q| = H/(dS)
    
    Parameters:
        H: numpy array lenght n containing values of H
        args: array of floats [beta, alpha, deltaT, k]

    Output:
        q: numpy array size n x 3 containing values of q given H at
        each equilibrium point

    To avoid dealing with the complexities of this please input values of H such that all radicands in dS are positive
    This can be done by finding where H*b / (k*a^2*dT^2) =< 1/4 (use the getH function below to create an array for H given the args)
    '''
    
    dS0 = args[1]*args[2] * (1/2 + np.sqrt(1/4 - H*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0] 
    dS1 = args[1]*args[2] * (1/2 - np.sqrt(1/4 - H*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0] 
    dS2 = args[1]*args[2] * (1/2 + np.sqrt(1/4 + H*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0]
    
    dS = [dS0, dS1, dS2]

    q0 = H / (dS[0])
    q1 = H / (dS[1])
    q2 = H / (dS[2])

    q = np.array([q0,q1,q2])

    return q

def getH(args, n):
    '''
    A function to determine the maximum value of H dS be non imaginary and then 
    create an array of n equally spaced values between 0 and the maximum H.

    Parameters:
    args: array of floats [beta, alpha, deltaT, k]
    n: length of returned H

    Outputs:
    H: numpy array of n equally spaced values of H between 0 and the calculated maximum value to avoid imaginary dS values
    '''

    Hmax = (args[3]*(args[1]**2)*(args[2]**2)) / (args[0]*4)
    H = np.linspace(0,Hmax,n)

    return H

def randQ(deltaS, args, rand, method):
    """
    Calculate q with various stochastic methods
    
    Parameters:
        deltaS: array, values to solve over
        args: array of form: [beta, alpha, deltaT, H, k]
            all elements are floats
        rand: randomly generated value
        method: 0,1,2,3, or 4 refers to which way of randomizing q to run
            0 is random addition to beta
            1 is random addition to alpha
            2 is random addition to deltaT
            3 is random addition to k
            4 is addative stochastic value at end
    """

    if method == 0: # random addition to beta
        q = args[4]*(args[1]*args[2] - (args[0]+rand)*deltaS)
    elif method == 1:
        q = args[4]*((args[1]+rand)*args[2] - args[0]*deltaS)
    elif method == 2:
        q = args[4]*(args[1]*(args[2]+rand) - args[0]*deltaS)
    elif method == 3:
        q = (args[4]+rand)*(args[1]*args[2] - args[0]*deltaS)
    else:
        q = args[4]*(args[1]*args[2] - args[0]*deltaS) + rand
    
    return q

def dDSnoArgs(deltaS, q, H):
    """
    Calculate dDS/dt from deltaS, q, and H rather than calculating q during the funciton (Needed for randomizing q)
    """
    return 2*H - 2*np.absolute(q)*deltaS