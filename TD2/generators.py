##Generators

##9.Fibonacci Sequence generator

def fibs():
    '''generator for fibonacci sequence'''
    r0, r1 = 0, 1
    while True:
        yield r0
        r0, r1 = r1, r0 + r1


def prefix_sums(k):
    '''generator to sum all integers from k to n'''
    k0, k1 = k, k+1
    while True:
        yield k0
        k0, k1 = k0+k1, k1+1



'''import random
def interleave(g1,g2):
    a=0
    k0, k1= random.choice(next(g1)), 1
    while True:
        yield k0
        k0, k1 = k1, random.choice(next(g1),next(g2))
g = interleave(prefix_sums(10), fibs())
print([next(g) for _ in range(10)])'''

'''
def choose(l,k):
    if k>len(l):
        return []
    if k==0:
        return [[]]
    if k==len(l):
        return [l]
    else:
        a=l[0]
        b=choose(l[1:],k-1)
        c=choose(l[1:],k)
        return [[a]+l for l in b]+[l for l in c]

'''

def choose_gen(l,k):
    if k>len(l):
        return
    if k==0:
         yield []
    elif k==len(l):
        yield l
    else:
        a=l[0]
        for i in choose_gen(l[1:],k-1):
            yield ([a]+i)
        for j in choose_gen(l[1:],k):
            yield j




