
def maze(m,n,i,j):
    if i==n-1 and j==n-1:
        return [(n-1,n-1)]
    else:
        if j<n-1 and m[i][j+1]!=0:
            path=maze(m,n,i,j+1)
            if path is not None:
                return [(i,j)]+path
        if i<n-1 and m[i+1][j]!=0:
            path=maze(m, n, i+1, j)
            if path is not None:
                return [(i,j)]+path



