# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def binom(n, k):
    if k > n:
        return 0
    elif k == 0 or k == n:
        return 1
    else:
        return binom(n - 1, k) + binom(n - 1, k - 1)


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
print(choose(list(range(4)),2))
def permutations(l):
    if not l:
        return [[]]
    a = []
    for i in l:
        temp = l[:]
        temp.remove(i)
        a.extend([i] + j for j in permutations(temp))
    return a


##choose and permutations without recursion
def choose2(l, k):
    if k == len(l):
        return l
    d = []
    for i in l:
        a = list(l)
        a.remove(i)
        d.append(a)
    for j in d:
        if len(j) > k:
            for m in j:
                b = list(j)
                b.remove(m)
                d.append(b)
    z = []
    for y in d:
        if y not in z:
            if len(y) == k:
                z.append(y)
    return z


print(choose2(list(range(10)), 5))
'''Obviously this is an extremely inefficient solution, but it worked first try'''


def permutations2(A, k):
    r = [[]]
    for i in range(k):
        r = [[a] + b for a in A for b in [i for i in r if a not in i]]
    return r


print(permutations2([1, 2, 3, 4], 4))


##multisets
def multichoose(S, k):
    pass


##not_angry

def not_angry(n):
    if n == 0:
        return 1
    if n == 1:
        return 2
    else:
        return not_angry(n - 1) + not_angry(n - 2)




