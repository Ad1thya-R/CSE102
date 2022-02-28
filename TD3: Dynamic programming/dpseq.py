def fibonacci(n):
    # Initially, r0 = F_0 & r1 = F_1
    r0, r1 = 0, 1

    for _ in range(n):
        # At each iteration, assuming that r0 contains F_n and
        # r1 contains F_(n+1), we can compute F_(n+2).
        #
        # We then "slide" the window s.t. after the assignments:
        #  - r0 contains F_(n+1)
        #  - r1 contains F_(n+2)
        r0, r1 = r1, r0+r1

        # We say that the window is of size 2 because we have two
        # variables (we could use a list of size 2 for example)

    # Starting from r0 = F_0 and r1 = F_1, after "n" iterations,
    # we know that r0 = F_n. We simply return that value.
    return r0

def next_seq(alphas, us):
    return sum([alphas[i]*us[i] for i in range(len(us))])



def u(alphas, us, n):
    seq=[alphas[i]*us[i] for i in range(len(us))]
    while len(seq)<=n:
        a = list(seq)
        seq.append(next_seq(alphas, a[len(seq)-len(us):len(seq)+1]))
    return seq[n]
