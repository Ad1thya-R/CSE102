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
from datetime import datetime
start_time_1=datetime.now()
def parts_td(n, k = None, cache = None):
    cache = {} if cache is None else cache
    if k is None:
        return sum(parts_td(n, k, cache) for k in range(1,n+1))
    if k==1:
        cache[n,k]=1
        return cache[n,k]
    if k>n:
        cache[n,k] = 0
        return cache[n,k]
    if (n, k) in cache:
        return cache[n, k]
    else:
        cache[n,k]=parts_td(n-1,k-1,cache)+parts_td(n-k,k,cache)
        return cache[n,k]
print(parts_td(500))


start_time = datetime.now()
print('Duration TD: {}'.format(start_time-start_time_1))
def parts_bu(n):
    cache = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    for m in range(n+1):
        for l in range(n+1):
            if l>m:
                break
            if l==1:
                cache[m][l]=1
            else:
                cache[m][l]=cache[m-1][l-1]+cache[m-l][l]
    return sum(cache[n])

print(parts_bu(500))

end_time = datetime.now()
print('Duration BU: {}'.format(end_time - start_time))