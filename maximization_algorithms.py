
from queue import Queue
import math
from heapq import heapify, heappush, heappop
import re

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

  def addEdge(self, u,v,data):
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
    ret = []
    for e in self.vertexMap.values():
      if len(e.keys()):
        ret.extend(list(e.keys()))
    return ret

  def getEdge(self,u,v):
    return self.vertexMap[u][(u,v)]

  def outgoing(self, v):
    return list(self.vertexMap[v].keys())

  def outdegree(self, v):
    return len(self.vertexMap[v])

  def incoming(self, v):
    return [(j,i) for (i,j) in self.vertexMap[v]]

  def indegree(self, v):
    return len(self.vertexMap[v])

  def str_path(self, v):
    ret = ""
    visited = set()
    visited.add(v)
    stack = []
    stack.append((v,None))
    while stack:
      (v, p) = stack.pop()
      if p:
        ret+=f"{p}--{self.getEdge(p,v)}--{v}  "

      for u in self.adjacents(v):
        if u not in visited:
          visited.add(u)
          stack.append((u,v))

    return ret.strip()

  def path(self, v):
    r = []
    adjacents = []
    visited = {v,}
    while True:
      for u in self.adjacents(v):
        if u not in visited:
          visited.add(u)
          adjacents.append((v, u))

      a, v = adjacents.pop()

      r.append(((a, v), self.getEdge(a, v)))

      if not adjacents:
        break

    return r #[row[0:1][0][1] for row in r[:]]

def dfs(G,v):
    t = set() # traveled
    p = [] # path
    dfs_rec(G, v, t, p)
    return p

def dfs_rec(G, v , t, p):
    t.add(v)
    for u in G.adjacents(v):
        if u not in t:
            p.append((v, u))
            dfs_rec(G, u, t, p)

class Vertex:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

def test():

    g = Graph()
    g.addVertex("a")
    g.addVertex("b")
    g.addVertex("c")
    g.addVertex("d")
    g.addVertex("e")
    g.addVertex("f")
    g.addVertex("g")
    g.addEdge("a","b",28)
    g.addEdge("a","c",10)
    g.addEdge("b","d",14)
    g.addEdge("c","f",25)
    g.addEdge("d","f",24)
    g.addEdge("d","g",18)
    g.addEdge("f","g",22)
    g.addEdge("g","e",12)
    g.addEdge("e","b",16)


    print(g.str_path('a'))
    print(dfs(g, 'a'))
    print(g.path('a'))
    exit(0)

    a = Vertex("a")
    b = Vertex("b")
    c = Vertex("c")
    d = Vertex("d")
    e = Vertex("e")
    f = Vertex("f")
    g = Vertex("g")

    g = Graph()
    g.addVertex(a)
    g.addVertex(b)
    g.addVertex(c)
    g.addVertex(d)
    g.addVertex(e)
    g.addVertex(f)
    g.addVertex(g)
    g.addEdge(a,b,28)
    g.addEdge(a,c,10)
    g.addEdge(d,f,25)
    g.addEdge(b,d,14)
    g.addEdge(b,e,16)
    g.addEdge(d,f,24)
    g.addEdge(d,g,18)
    g.addEdge(f,g,22)
    g.addEdge(g,e,12)

    path_from_a = g.path(a)
    print(path_from_a) # até aqui funciona
    try:
        print(a in path_from_a) # não funciona
    except TypeError:
        print('Path é uma string, portanto só é possível localizar outra string dentro dele')
    print('a' in 'a--b a--c') # Se o vértice for uma string, ok, mas a classe Graph não deveria aceitar qualquer tipo como vértice, como demostrado,
                              # ou a funcão path() não deveria retornar uma string


def exists_on_path(t, i, x):
    p = t.path(i)
    vertices = re.findall(r"[a-zA-Z']+", p)
    return x in vertices

def kruskals(g):
    edges = [((i, j), g.getEdge(i, j)) for (i, j) in g.edges()]
    edges.sort(key=lambda e: e[1])

    t = Graph()

    while edges:
        e, data = edges.pop(0)
        u, v = e[0], e[1]

        if u not in t.vertices() and v not in t.vertices():
            t.addVertex(u)
            t.addVertex(v)
            t.addEdge(u, v, data)
        else:
            if u in t.vertices() and v in t.vertices():
               if not exists_on_path(t, u, v):
                   t.addEdge(u, v, data)
            elif u in t.vertices() and v not in t.vertices():
                t.addVertex(v)
                t.addEdge(u, v, data)
#    print(t.path('a'))
    return t


def get_min_edge(g):
    min_edge = None
    for e in g.edges():
        data = g.getEdge(e[0], e[1])
        if not min_edge:
            min_edge = (e, data)
        else:
            if data < min_edge[1]:
                min_edge = (e, data)
    return min_edge

def get_min_adjacent(g, t):
    min_adjacent = None
    for v in t.vertices():
        adjacents = [((i, j), g.getEdge(i, j)) for (i, j) in g.outgoing(v)]
        for a in adjacents:
            if a[0][1] not in t.vertices():
                if not min_adjacent:
                    min_adjacent = a
                else:
                    if a[1] < min_adjacent[1]:
                        min_adjacent = a
    return min_adjacent

def prims(g):
    t = Graph()
    e, data = get_min_edge(g)
    t.addVertex(e[0])
    t.addVertex(e[1])
    t.addEdge(e[0], e[1], data)

    t_edges_counter = 1
    while t_edges_counter < len(g.vertices()) - 1:
        for v in t.vertices():
            r = get_min_adjacent(g, t)
            if r:
                a, data = r
                t.addVertex(a[1])
                t.addEdge(a[0], a[1], data)
                t_edges_counter += 1

    # print(t.path('a'))
    return t

def run():
    # Graph 1
    g = Graph()
    g.addVertex("a")
    g.addVertex("b")
    g.addVertex("c")
    g.addVertex("d")
    g.addVertex("e")
    g.addVertex("f")
    g.addVertex("g")
    g.addEdge("a","b",28)
    g.addEdge("a","c",10)
    g.addEdge("c","f",25)
    g.addEdge("b","d",14)
    g.addEdge("b","e",16)
    g.addEdge("d","f",24)
    g.addEdge("d","g",18)
    g.addEdge("f","g",22)
    g.addEdge("g","e",12)

    p = prims(g)

    assert sorted(p.vertices()) == ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    assert sorted(p.edges()) == sorted([('a', 'c'), ('b', 'e'), ('b', 'd'), ('c', 'a'), ('c', 'f'), ('d', 'b'), ('e', 'g'), ('e', 'b'), ('f', 'c'), ('f', 'g'), ('g', 'f'), ('g', 'e')])
    assert p.path("a") == "a--10--c  c--25--f  f--22--g  g--12--e  e--16--b  b--14--d"

    p = kruskals(g)

    assert sorted(p.vertices()) == ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    assert sorted(p.edges()) == sorted([('a', 'c'), ('b', 'e'), ('b', 'd'), ('c', 'a'), ('c', 'f'), ('d', 'b'), ('e', 'g'), ('e', 'b'), ('f', 'c'), ('f', 'g'), ('g', 'f'), ('g', 'e')])
    assert p.path("a") == "a--10--c  c--25--f  f--22--g  g--12--e  e--16--b  b--14--d"

    # Graph 2
    g = Graph()
    g.addVertex("a")
    g.addVertex("b")
    g.addVertex("c")
    g.addVertex("d")
    g.addVertex("e")
    g.addVertex("f")
    g.addVertex("g")
    g.addVertex("h")
    g.addEdge("a","b",6)
    g.addEdge("a","c",4)
    g.addEdge("b","c",5)
    g.addEdge("b","e",14)
    g.addEdge("b","h",10)
    g.addEdge("c","f",2)
    g.addEdge("c","d",9)
    g.addEdge("e","h",3)
    g.addEdge("f","g",15)
    g.addEdge("f","h",8)

    p = prims(g)
    assert sorted(p.vertices()) == ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    assert sorted(p.edges()) == sorted([('a', 'c'), ('b', 'c'), ('c', 'f'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('d', 'c'), ('e', 'h'), ('f', 'c'), ('f', 'h'), ('f', 'g'), ('g', 'f'), ('h', 'f'), ('h', 'e')])
    assert p.path("a") == "a--4--c  c--9--d  c--5--b  c--2--f  f--15--g  f--8--h  h--3--e"

    p = kruskals(g)
    assert sorted(p.vertices()) == ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    assert sorted(p.edges()) == sorted([('a', 'c'), ('b', 'c'), ('c', 'f'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('d', 'c'), ('e', 'h'), ('f', 'c'), ('f', 'h'), ('f', 'g'), ('g', 'f'), ('h', 'f'), ('h', 'e')])
    assert p.path("a") == "a--4--c  c--9--d  c--5--b  c--2--f  f--15--g  f--8--h  h--3--e"

    print("Here comes the sun...")


if __name__ == "__main__":
    test()
