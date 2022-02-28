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
print(binom_td(400, 100),i)
