def required(G,c, visited=[]):
    if c not in visited:
        visited.append(c)
        for neighbour in G[c]:
            required_list(G,neighbour,visited)
    return len(visited)


def required_list(G,c, visited=[]):
    if c not in visited:
        visited.append(c)
        for neighbour in G[c]:
            required_list(G,neighbour,visited)
    return visited

def revert_edges(G):
    ans = [[] for _ in G]
    for i, l in enumerate(G):
        for x in l:
            ans[x].append(i)
    return ans

def needed_for(G,c):
    rev=revert_edges(G)
    return required(rev, c)



