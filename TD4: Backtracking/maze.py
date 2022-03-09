
def maze(m,n,i,j):
    if m[i][j]==0:
        return
    elif i==n-1 and j==n-1:
        return [(n-1,n-1)]
    else:
        if j<n-1 and m[i][j+1]!=0 and m[i][j+1] is not None:
            return [(i,j)]+maze(m,n,i,j+1)
        if i<n-1 and m[i+1][j]!=0 and m[i+1][j] is not None:
            return [(i,j)]+maze(m, n, i+1, j)
        else:
            return None



