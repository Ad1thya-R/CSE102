import random

def is_prime(n, k=32):
    '''
    Parameters
    ----------
    n : strictly positive natural number
        The number that you are checking whether it is a prime or not
    k : integer
        k controls the accuracy of the test. The default is 32.

    Returns whether number is prime based on Rabin Miller algorithm
    -------
    '''
    if n <= 3:
        return n == 2 or n == 3
    if n%2==0:
        return False
    r = 0
    d = (n - 1)
    while d % 2 == 0:
        d //= 2
        r += 1
    d = int(d)
    for i in range(k):
        a = random.randint(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for j in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def genprime(l):
    '''
    :param l:Specify the number l of bits which the prime has to be
    :return: a prime number of approximately l bits.
    '''
    num=random.getrandbits(l)
    num= num | (1 << 1)
    if num%2==0:
        num+=1
    prime=num
    i=0
    while not is_prime(prime):
        i+=1
        prime+=2 * i
    return prime

def egcd(b, a):
    x0, x1, y0, y1 = 1, 0, 0, 1
    while a != 0:
        q, b, a = b // a, a, b % a
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return b, x0, y0

def genmod(p,q):
    '''

    :param p: prime number
    :param q: prime number
    :return: RSA public and secret key pair is returned
    '''






