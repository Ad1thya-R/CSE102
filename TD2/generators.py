##Generators

##9.Fibonacci Sequence generator

def fibs():
    '''generator for fibonacci sequence'''
    r0, r1 = 0, 1
    while True:
        yield r0
        r0, r1 = r1, r0 + r1