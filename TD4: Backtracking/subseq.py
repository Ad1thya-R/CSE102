'''
def bt(c):
    # "c" is a partial solution to our problem
    if reject(c):
        # If "c" cannot be completed further, we prune all
        # the children of "c" -- i.e. we simply return from
        # the call to "bt(c)"
        return
    elif accept(c):
        # If "c" is a valid complete solution, we report it
        yield c
    else:
        # Otherwise, we iterate over all the partial solutions
        # that can be obtained from a single modification to "c"
        #
        # For each of these sub-partial solution, we do a
        # recursive call to "bt"
        for subc in children(c):
            yield from bt(subc)
'''
def subseq(seq):
    '''Given a sequence as an array, write a generator
    that returns all subsequences of the sequence as arrays
    in some arbitrary order'''
    subseq=[]
    res=[]
    def dfs(i):
        if i >= len(seq):
            res.append(subseq[:])
            return
        subseq.append(seq[i])
        dfs(i+1)

        subseq.pop()
        dfs(i+1)
    dfs(0)
    return res

def subseq_2(seq):
    """ Given a list of elements return a generator
    that will generate all the subsets """

    if len(seq) == 1:
        yield seq
        yield []
    else:
        for item in subseq_2(seq[1:]):
                yield [seq[0]] + item
                yield item
print([s for s in subseq([1,2,3,4])])

def subseq_personal(seq):

