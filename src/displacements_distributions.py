import numpy as np
from scipy import signal

def qExponential(q,x):
    """ Returns a discrete Q-exponential probability distribution, defined by 
    the q parameter.
    """
    if q == 1:
        probability_distribution = (2-q)*np.exp(-x)

    elif q < 1 and q >=0:
        probability_distribution = []
        if q == 0.5:
            return signal.unit_impulse(np.shape(x),0)
        for i in x:
            if (1-q)*i <= 1:
                probability_distribution.append((2-q) * (1-(1-q)*i)**(1/(1-q)))
            else:
                probability_distribution.append(0)

    elif q > 1 and q <= 10**(3):
        probability_distribution = (2-q)*(1-(1-q)*x)**(1/(1-q))

    elif q > 10**(3):
        probability_distribution = np.ones((np.size(x)))
    
    normalization = sum(probability_distribution)
    
    return probability_distribution/normalization

def correlated_displacements(rho, l_x, l_a):
    """ Function that introduces a correlation between the steps sizes
    in l_x and l_y.

    rho (float in [-1,1] interval): parameter that specifies the level of the
    correlation;

    l_x: displacements array in which we want to correlate with;

    l_a: independently generated displacement array accordingly 
         with the y direction parameters.

    The displacements array must be of the same size.
    """

    l_x = np.array(l_x)
    l_a = np.array(l_a)

    l_y = rho*l_x + np.sqrt(1 - rho**2)*l_a

    return np.array([max(dy,0) for dy in l_y])


# Dictionary used to call the above functions through the specification in the .cfg file
functions = {'qExponential': qExponential, 
            'correlated_displacements': correlated_displacements}
