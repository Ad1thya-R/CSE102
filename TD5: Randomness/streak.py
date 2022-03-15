import random

def prob(n,k):
    '''Suppose you flip a coin N successive times.
    What is the probability that there was a streak of k heads? '''
    sk=0
    win=0
    for _ in range(n):
        s=random.randint(0,1)
        if s==1:
            sk+=1
            if sk>=k:
                win=1
        if s==0:
            if sk>=k:
                win=1
            sk=0

    return win == 1

def experiment(n,k,t=10000):
    w=0
    for i in range(t):
        if prob(n,k)==1:
            w+=1
    return w/t

print(experiment(30,7))