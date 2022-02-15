##Generators

##9.Fibonacci Sequence generator

def fibs():
    '''generator for fibonacci sequence'''
    r0, r1 = 0, 1
    while True:
        yield r0
        r0, r1 = r1, r0 + r1
g=fibs()
print([next(g) for _ in range(10)])

def prefix_sums(k):
    '''generator to sum all integers from k to n'''
    k0, k1 = k, 2*k+1
    while True:
        yield k0
        k0, k1 = k1, 2*k1+1

g = prefix_sums(10)
print([next(g) for _ in range(10)])

'''import random
def interleave(g1,g2):
    a=0
    k0, k1= random.choice(next(g1)), 1
    while True:
        yield k0
        k0, k1 = k1, random.choice(next(g1),next(g2))
g = interleave(prefix_sums(10), fibs())
print([next(g) for _ in range(10)])'''
lista = []
iteraciones = [0]

'''
def choose(l, k):
    if k == len(l):
        if not l in lista:
            lista.append(l)
        return
    for i in l:
        aux = l[:]
        aux.remove(i)
        result = choose(aux, k)
        iteraciones[0] += 1
        if not result in lista and result:
            lista.append(result)
    print(lista)
'''

def choose_gen(l: object, k: object) -> object:
    if k>len(l):
        return None
    if k == len(l):
        yield sorted(l)
        return
    for i in l:
        aux = l[:]
        aux.remove(i)
        result = choose_gen(aux, k)
        if result:
                yield from result
choose([1,3,5,7], 2)
g = choose_gen(list(range(100)), 50)
a=2


