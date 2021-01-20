from heapq import heappush, heappop, heapify
from queue import Queue

class Graph:
  def __init__(self):
    self.vertexMap = dict()

  def addVertex(self, v):
    self.vertexMap[v] = dict()

  def removeVertex(self, v):
    if v in self.vertexMap:
      for (i,j) in self.vertexMap[v].copy():
        print(f"e->{(i,j)}")
        self.removeEdge(i,j)
      del self.vertexMap[v]

  def vertices(self):
    return list(self.vertexMap.keys())

  def adjacents(self, v):
    return [j for (i, j) in self.outgoing(v)]

  def addEdge(self,u,v,data):
    if (u in self.vertexMap) and (v in self.vertexMap):
      self.vertexMap[u][(u,v)] = data
      self.vertexMap[v][(v,u)] = data
    else:
      raise ValueError(f"One or both of the V {u} and {v} are not present in the Graph!")

  def removeEdge(self,u,v):
    if ((u,v) in self.vertexMap[u]) and ((v,u) in self.vertexMap[v]):
      del self.vertexMap[u][(u,v)]
      del self.vertexMap[v][(v,u)]

  def edges(self):
    return [list(e.keys()) for e in self.vertexMap.values() if len(e.keys())]

  def getEdge(self,u,v):
    return self.vertexMap[u][(u,v)]

  def dist(self, u, v):
    return self.getEdge(u, v)

  def outgoing(self, v):
    return list(self.vertexMap[v].keys())

  def outdegree(self, v):
    return len(self.vertexMap[v])

  def incoming(self, v):
    return [(j,i) for (i,j) in self.vertexMap[v]]

  def indegree(self, v):
    return len(self.vertexMap[v])

def dfs(G,v):
    t = set() # traveled
    p = [] # path
    dfs_rec(G, v, t, p)
    return p

def dfs_rec(G, v , t, p):
    t.add(v)
    for u in G.adjacents(v):
        if u not in t:
            p.append(u)
            dfs_rec(G, u, t, p)

def bfs(G,v):
    p = [] # path
    q = Queue()
    q.put_nowait(v)
    t = {} # traveled
    t.add(v)
    while q.qsize():
        e = q.get_nowait()
        for u in G.adjacents(e):
            if u not in t:
                t.add(u)
                q.put_nowait(u)
                p.append(u)
    return p

def shortestPath(v, u, paths):
    l = []
    while v != u:
        (d, p) = paths[u]
        l.insert(0, u)
        u = p
    l.insert(0, u)
    return l

def dijkstra(G,v,u):
    paths = {} # paths
    Q = []
    h = heapify(Q)
    heappush(Q, (0, v, None))
    while len(Q):
        d_xv, x, p = heappop(Q)
        if x == u:
            paths[x] = (d_xv, p)
            return shortestPath(v, u, paths)
        if x not in paths.keys():
            paths[x] = (d_xv, p)
            for t in G.adjacents(x):
                d_vt = d_xv + G.getEdge(x, t)
                if t not in paths.keys() or d_vt < paths[t][0]:
                    heappush(Q, (d_vt, t, x))

def run():
    g = Graph()
    g.addVertex("a")
    g.addVertex("b")
    g.addVertex("c")
    g.addVertex("d")
    g.addVertex("e")
    g.addVertex("f")
    g.addVertex("g")
    g.addVertex("h")
    g.addVertex("i")
    g.addVertex("j")
    g.addVertex("k")
    g.addVertex("l")
    g.addVertex("m")
    g.addVertex("n")
    g.addVertex("o")
    g.addVertex("p")
    g.addEdge("a","b",1)
    g.addEdge("a","e",1)
    g.addEdge("a","f",1)
    g.addEdge("b","c",1)
    g.addEdge("b","f",1)
    g.addEdge("c","d",1)
    g.addEdge("c","g",1)
    g.addEdge("d","g",1)
    g.addEdge("d","h",1)
    g.addEdge("e","f",1)
    g.addEdge("e","i",1)
    g.addEdge("f","i",1)
    g.addEdge("g","l",1)
    g.addEdge("g","k",1)
    g.addEdge("g","j",1)
    g.addEdge("h","l",1)
    g.addEdge("i","m",1)
    g.addEdge("i","n",1)
    g.addEdge("i","j",1)
    g.addEdge("j","k",1)
    g.addEdge("k","o",1)
    g.addEdge("l","p",1)
    g.addEdge("m","n",1)
    g.addEdge("n","k",1)

    # assert dfs(g,'a') == ['b', 'c', 'd', 'g', 'l', 'h', 'p', 'k', 'j', 'i', 'e', 'f', 'm', 'n', 'o']

    # assert bfs(g,'a') == ['b', 'e', 'f', 'c', 'i', 'd', 'g', 'm', 'n', 'j', 'h', 'l', 'k', 'p', 'o']

    assert dijkstra(g, 'a','g') == ['a', 'b', 'c', 'g']

    assert dijkstra(g, 'a','j') == ['a', 'e', 'i', 'j']

    assert dijkstra(g, 'a','l') == ['a', 'b', 'c', 'g', 'l']

    assert dijkstra(g, 'a','i') == ['a', 'e', 'i']

    print("Parabéns!!! Atividade 5 concluída com sucesso!")

if __name__ == "__main__":
    run()
