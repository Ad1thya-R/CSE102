import random
import math

def torus_volume_cuboid(R, r, N=100_000):
    M=0
    for _ in range(N):
        x= random.uniform(-R-r,R+r)
        y= random.uniform(-R -r,R+r)
        z = random.uniform(-r, r)
        if (math.sqrt(x**2+y**2)-R)**2+z**2<=r**2:
            M+=1
    A=2*r
    B=2*(R+r)
    return M/N * A * B**2




