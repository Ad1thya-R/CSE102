import random


def penney():
    streak=[random.randint(0,1) for _ in range(3)]
    if [streak[0], streak[1], streak[2]] == [1, 1, 0]:
        return 1
    if [streak[0], streak[1], streak[2]] == [0, 1, 1]:
        return 0
    while [streak[0],streak[1],streak[2]]!=[1,1,0] and [streak[0],streak[1],streak[2]]!=[0,1,1]:
        x=random.choice([0,1])
        streak[0],streak[1],streak[2]=streak[1],streak[2],x
        if [streak[0],streak[1],streak[2]]==[1,1,0]:
            return 1
        if [streak[0],streak[1],streak[2]]==[0,1,1]:
            return 0
def experiment(t=10000):
    w=0
    for _ in range(t):
        w+=penney()
    return w/t

print(experiment())