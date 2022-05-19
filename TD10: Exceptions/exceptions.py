class InvalidInput(Exception):
    pass

def sum_of_input():
    '''Takes user input of numders (each on a separate line),
    and then we return the su of the inputs'''
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        try:
            num = int(line)
        except ValueError:
            raise InvalidInput
        contents.append(num)
    return sum(contents)

class Found(Exception):
    def __init__(self, solution):
        self.solution = solution


def subset_sum(nm,S,M):
    '''Take the subset_sum backtracking algorithm from Lecture 4 and
    modify it s.t. it raises an exception Found, that you have to
    define, when a solution is found. This exception should have an
    attribute solution that contains the solution to the subset sum problem.'''
    # "nm" is the set of available numbers
    # "M" is the target sum
    # "S" is the current partial solution

    nS = sum(S)  # The sum of the partial solution

    if nS > M:
        # "S" is a non-feasible solution.
        # We reject it.
        return None

    if nS == M:
        # S is a valid solution.
        # We accept it.
        return S

    for i in nm:
        # Otherwise, we try to extend S with the integers
        # from "nm" - 1 by 1 and continue recursively.
        rS = subset_sum(
            nm.difference([i]),  # We remove "i" from "nm"
            S.union([i]),  # We add "i" to "S"	input_column_len=len(C)
            M  # The targeted sum is unchanged
        )

        if rS is not None:
            # We found a solution (recursively)
            # We return it
            raise Found(rS)

    # We tried all the numbers in "nm" without finding
    # a solution. We report the failure with "None"
    return None
