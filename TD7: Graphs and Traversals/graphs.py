def matrix_to_adjlist ( G ):
    n = len ( G )
    L = []
    for i in range ( n ):
        L . append ([])
        for j in range ( n ):
            if G [ i ][ j ]:
                L [ i ]. append ( j )
    return L

def is_symmetric(G):
    n=len(G)
    for i in range(n):
        if G[i]!=[]:
            for j in range(len(G[i])):
                if i not in G[G[i][j]]:
                    return False
    return True

def revert_edges(G):
    ans = [[] for _ in G]
    for i, l in enumerate(G):
        for x in l:
            ans[x].append(i)
    return ans

G=[
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
M=matrix_to_adjlist(G)
print(M)
print(is_symmetric(M))
M2=[[],[4,3],[],[1],[1],[]]
print(is_symmetric(M2))
print(revert_edges(M))
