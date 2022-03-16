import random


def roll(D):
    randRoll = random.random()
    sum = 0
    result = 1
    for mass in D:
        sum += mass
        if randRoll < sum:
            return result
        result += 1

def rolls(D,N):
    rolls=[0 for _ in range(len(D))]
    for _ in range(N):
        k=roll(D)-1
        rolls[k]+=1
    return tuple(rolls)

import matplotlib.pyplot as plt


def plot(ns):
    N  = sum(ns)
    ns = [float(x) / N for x in ns]
    plt.bar(range(len(ns)), height=ns)
    plt.xticks(range(len(ns)), [str(i+1) for i in range(len(ns))])
    plt.ylabel('Probability')
    plt.title('Biased die sampling')
    plt.show()

print(rolls([0.1,0.3,0.2,0.4],10000))

