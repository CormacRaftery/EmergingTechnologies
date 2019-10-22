# Adapted from https://tour.golang.org/flowcontrol/8

def sqrt(x):
    #initial guess
    z=1.0
    #guess until within 4 decimals
    while abs(z*z -x) >= 0.0001:
        #best approximation
        z-=(z*z-x)/(2*z)
    return z
z = sqrt(63.0)
print(z)
print(z*z)