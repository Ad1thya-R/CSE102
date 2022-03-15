import random


def penney():
    streak=[random.randint(0,1),random.randint(0,1)]
    for i in range(2,10000):
        s=random.randint(0,1)
        streak.insert(i,s)
        if [streak[i-2],streak[i-1],streak[i]]==[1,1,0]:
            return 1
        if [streak[i-2],streak[i-1],streak[i]]==[0,0,1]:
            return 0

def experiment(t=10000):
    w=0
    for _ in range(t):
        w+=penney()
    return w/t
print(experiment())
