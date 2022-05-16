import numpy as np
def float_to_int(x):
    '''takes a float input and returns the integer part'''
    return int(str(x)[0])
def uint16_to_bitstring(x):
    '''convert decimal to binary with int() function'''
    bitstring = [0] * 16
    while x>0:
        k = float_to_int(np.log2(x))
        bitstring[15-k]=1
        x -= 2**k
    return bitstring
'''def unit16_to_bitstring(x):
    Converts denary to binary representation under assumption that x<2**16
    if x>1:
        return unit16_to_bitstring(x//2)+unit16_to_bitstring(x%2)
    else:
        return [x]
'''
print(uint16_to_bitstring(3905))

def bitstring_to_uint16(b):
    '''Converts binary to denary representation under assumption that b<2**16'''
    x=0
    for i, bit in enumerate(b):
        if bit==1:
            x+=2**(15-i)

    return x


def mod_pow2(x, k):
    '''bitwise implementation of x%2**k'''
    return x & (2**k-1)


def is_pow2(x):
    '''is x a power of 2'''
    return x & (x-1) == 0 and x > 0

def set_mask(w,m):
    return w | m

def toggle_mask(w,m):
    return w ^ m

def clean_mask(w,m):
    return ~(~w ^ m)

def tap_uint16(x, i):
    """ Return 1 or 0 depending on whether the ith-least significant bit
        of x is 1 or 0.
    """
def slowtap_uint16(x, i):
    return uint16_to_bitstring(x)[15-i]
print(slowtap_uint16(42,1))




