def binom_td(n, k, cache=None):
    global i
    i+=1
    cache = {} if cache is None else cache
    if (n, k) not in cache:
        if k == 0 or k == n:
            cache[n, k] = 1
        else:
            cache[n, k] = binom_td(n - 1, k - 1, cache) + binom_td(n - 1, k, cache)
    return cache[n, k]

i=0
print(binom_td(200, 100),i)

def parts_td(n, k = None, cache = None):
    cache = {} if cache is None else cache
    if k is None:
        return 1
    if (n, k) in cache:
        return cache[n, k]
    else:
        cache[n,k]=parts_td(n-1,k-1,cache)+parts_td(n-k,k,cache)



print([parts_td(n) for n in range(1,10)])