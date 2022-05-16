def tap_uint16(x, i):
    """ Return 1 or 0 depending on whether the ith-least significant bit
        of x is 1 or 0.
    """
    newnum= (x >> i)
    return newnum & 1

def polytap_uint16(x, I):
    """ Tap x at all the positions in I (which is a list of tap
        positions) and return the xor of all of these tapped values.
    """
    tapped=[]
    for i in I:
        tapped.append(tap_uint16(x,i))
    w=tapped[0]
    for t in range(1, len(tapped)):
        w = w ^ tapped[t]
    return w

def lfsr_uint16(x, I):
    """Return the successor of x in an LFSR based on tap positions I"""
    r = x >> 1
    """set the 15th bit of r to polytap_uint16(x,I)"""
    if polytap_uint16(x,I)==1:
        return r | (1 << 15)
    else:
        return r & (~ (1 << 15))


print(polytap_uint16(3905,[2,4,6]))

