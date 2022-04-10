def required(G,c, visited=[]):
    if c not in visited:
        visited.append(c)
        for neighbour in G[c]:
            required(G,neighbour,visited)
    return len(visited)

print(required([[2], [0], []], 0))