def shortest_route_len(G, s, t, visited=[], queue=[]):
  visited.append(s)
  queue.append(s)

  while queue:
        s = queue.pop(0)

        for neighbour in G[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
  return len(visited)-2


print(shortest_route_len([[1], [], [0, 1]], 0, 0))
