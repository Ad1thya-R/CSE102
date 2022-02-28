def catalan(n):
    '''Compute caalan numbers using only recursion (inefficient approach)'''
    if n==0:
        return 1
    else:
        return sum([catalan(i)*catalan(n-1-i) for i in range(n)])

def catalan_td(n, cache = None):
    cache = {} if cache is None else cache
    if n not in cache:
        # This is the first time we compute `catalan(n)`
        # Compute it and store the result in `cache[n]`
        if n == 0:
            cache[n]=1
        else:
            cache[n]=sum([catalan_td(i, cache) * catalan_td(n - 1 - i, cache) for i in range(n)])
    # At that point, we know that `cache[n]` exists and
    # is exactly the n-th Catalan number
    #
    # We simply return it.
    return cache[n]


print(catalan_td(50))

def next_catalan(cs):
    n=len(cs)
    if n==0:
        return 1
    else:
        return sum([cs[i]*cs[n-1-i] for i in range(n)])

def catalan_bu(n):
    if n==0:
        return 1
    else:
        cs=[]
        while len(cs)<=n:
            csa=list(cs)
            cs.append(next_catalan(csa))
    return cs[n]


