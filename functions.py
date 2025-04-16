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

def getH(args, n, start=False):
    '''
    A function to determine the maximum value of H dS be non imaginary and then 
    create an array of n equally spaced values between 0 and the maximum H.

    Parameters:
    args: array of floats [beta, alpha, deltaT, k]
    n: length of returned H
    start: where to start the list from, (defaults to -Hmax), if start > Hmax will return the list from Hmax to start

    Outputs:
    H: numpy array of n equally spaced values of H between 0 and the calculated maximum value to avoid imaginary dS values
    '''

    Hmax = (args[3]*(args[1]**2)*(args[2]**2)) / (args[0]*4)

    if start == False:
        H = np.linspace(-Hmax, Hmax,n)
    elif start < Hmax:
        H = np.linspace(start,Hmax,n)
    elif start > Hmax:
        H = np.linspace(Hmax,start,n)

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

def q_H2(args, start, end, n):
    '''
    A Function of q in terms of H at equilibrium points:
        |q|dS = H => |q| = H/(dS)
    
    Parameters:
        args: array of floats [beta, alpha, deltaT, k]
        start: lowest value of H wanted, note if start < end, values will be swapped
        end: highest value of H wanted, note if start < end, values will be swapped
        n: number of H's wanted (this will be the sum of the length of q0 (or q1) and q2), must be even

    Output:
        q0: array size 2 by i containing values of q within the first equilibrium zone: q0[0] = qi0 ; q0[1] = qd0 (note i+j=n)
        q1: array size 2 by i containing values of q within the second equilibrium zone: q1[0] = qi1 ; q1[1] = qd1 (note i+j=n)
        q2: array size 2 by j containing values of q within the third equilibrium zone: q2[0] = qi2 ; q2[1] = qd2 (note i+j=n)
        time: 6 arrays of times
        H: the 4 arrays of H values used to calculate q
    '''

    if start > end:
        start, end = end, start
    
    Hmax = (args[3]*(args[1]**2)*(args[2]**2)) / (args[0]*4) # maximum value of H such that dS0 and dS1 are real (note that -Hmax is the minimum such that dS2 is real)

    # H Increasing
    Hi = np.linspace(start,end,int(n / 2)) # H increasing
    Hi1 = Hi[np.where(Hi <= Hmax)] # All areas where first eq and unstable eq are real
    Hi2 = Hi[np.where(Hi >= -Hmax)] # All areas where third eq is real
    
    dSi0 = args[1]*args[2] * (1/2 - np.sqrt(1/4 - Hi1*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0] # First stable equilibrium zone
    timei0 = np.arange(len(Hi1)) # Times for dSi0 and dSi1 are just from start till the index where Hi reaches Hmax (for simplicity this can be expressed as just the length of Hi1)

    dSi1 = args[1]*args[2] * (1/2 + np.sqrt(1/4 - Hi1*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0] # Unstable equilibrium zone
    timei1 = np.arange(len(Hi1)) # Times for dSi0 and dSi1 are just from start till the index where Hi reaches Hmax (for simplicity this can be expressed as just the length of Hi1)

    dSi2 = args[1]*args[2] * (1/2 + np.sqrt(1/4 + Hi2*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0] # Second stable equilibrium zone
    timei2 = np.arange((n/2)-len(Hi2), (n/2)) # Time for dSi2, from the index where Hi passes -Hmax until the end of Hi (in terms of math this starts at n/2 - the length of Hi2 and continues until Hi index at n/2)

    qi0 = Hi1 / (dSi0) # q in first stable equilibrium zone
    qi1 = Hi1 / (dSi1) # q in  unstable equilibrium zone
    qi2 = Hi2 / (dSi2) # q in second stable equilibrium zone

    # H Decreasing

    Hd = np.linspace(end,start,int(n / 2)) # H decreasing
    Hd1 = Hd[np.where(Hd <= Hmax)] # All areas where first eq and unstable eq are real
    Hd2 = Hd[np.where(Hd >= -Hmax)] # All areas where third eq is real
    
    dSd0 = args[1]*args[2] * (1/2 - np.sqrt(1/4 - Hd1*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0] # First stable equilibrium zone
    timed0 = np.arange(n - len(Hd1),n) # Times for dSd0 and dSd1 are just from the index Hd drops below Hmax till until n (from n/2 + length of Hd1 to n)

    dSd1 = args[1]*args[2] * (1/2 + np.sqrt(1/4 - Hd1*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0] # Unstable equilibrium zone
    timed1 = np.arange(n - len(Hd1),n) # Times for dSd0 and dSd1 are just from the index Hd drops below Hmax till until n (from n/2 + length of Hd1 to n)

    dSd2 = args[1]*args[2] * (1/2 + np.sqrt(1/4 + Hd2*args[0]/(args[3]*(args[1]**2)*(args[2]**2)))) / args[0] # Second stable equilibrium zone
    timed2 = np.arange(n/2, n/2+len(Hd2)) # Times for dSd2 is from n/2 until the index where Hd drops below -Hmax (n/2 until n/2 + length of Hd2)


    qd0 = Hd1 / (dSd0) # q in first stable equilibrium zone
    qd1 = Hd1 / (dSd1) # q in  unstable equilibrium zone
    qd2 = Hd2 / (dSd2) # q in second stable equilibrium zone

    # Returning Calculated Values
    H = [Hi1,Hi2,Hd1,Hd2]
    times = [timei0, timed0, timei1, timed1, timei2, timed2]
    q0 = [qi0,qd0]
    q1 = [qi1,qd1]
    q2 = [qi2,qd2]

    return q0, q1, q2, times, H