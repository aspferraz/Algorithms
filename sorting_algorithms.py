import numpy as np
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
import timeit
from joblib import Parallel, delayed


def swap(a, i1, i2):
    a[i1], a[i2] = a[i2], a[i1]


def hoaresPartition(a, l, h):
    i, j, p = l, h - 1, a[h]

    while True:
        while a[i] <= p and i < j: # move forward with i
            i += 1

        while a[j] >= p and i < j: # move backwards with j
            j -= 1

        if i == j: # when the indexes meet
            if a[i] <= p: 
                i += 1

            swap(a, i, h)
            
            return i # return the pivot index

        else: # while the indexes have not yet met themselves
            swap(a, i, j)


def lomutosPartition(a, l, h):
    #if l != h:
    #   pIdx = random.randint(l, h) # choose random pivot
    #   a[h], a[pIdx] = a[pIdx], a[h] # moves pivot to end of the list

    
    p = a[h] # select the last element to be the pivot 

    s = l # split index starts with the left most
    for i in range(l, h):
        if a[i] <= p: 
            swap(a,i,s) # swap i with s if value in i were less or equal than pivot 
            s += 1 # move forward split index
    
    swap(a,s,h)

    return s # return the pivot index


# Brute force, O(n^2)
def insertionSort(a):
    j = 1
    while j < len(a):
        for i in range(j):
            if a[j] < a[i]:
                swap(a, i, j)
        j += 1


# Brute force, O(n^2)
def insertionSortWithSentinel(a):
    # get the index of the lowest value of the array, O(n)
    s = 0
    for i in range(1, len(a)):
        if a[i] < a[s]:
            s = i      

    swap(a, 0, s) # sentinel is placed at the beginning of the array to simplify the inner loop

    for j in range(2, len(a)):
        i = j - 1
        t = a[j]
        
        while t < a[i]:
            swap(a, i + 1, i)
            i -= 1   
          
        a[i + 1] = t
      


# Brute force, O(n^2)
def bubbleSort(a):
    while True:
        notSwapped = True
        for i in range(len(a) - 1):
            if (a[i] > a[i + 1]):
                swap(a, i, i + 1)
                notSwapped = False
        if notSwapped:
            break


# Brute force, O(n^2)
def selectionSort(a):
    for i in range(len(a) - 1):
        u = i # lowest unordered index
        for j in range(i + 1, len(a)):
            if (a[j] < a[u]):
                u = j
        swap(a, u, i)


def heapify(a, end, root):      
    hi = root           # current highest index
    l = 2 * root + 1    # left child
    r = 2 * root + 2    # right child
  
    # if exists child node at right and it is greater than root
    if r < end and a[r] > a[hi]: 
        hi = r 

    # if exists child node at left and it is greater than root
    if l < end and a[l] > a[hi]: 
        hi = l 

    # if root is not the highest, swap and continue heapifying
    if root != hi: 
        swap(a, root, hi) # swap
        heapify(a, end, hi) 

  
def heapSort(a): 
    n = len(a) 
    
    # creates a max-heap
    start = n // 2 - 1 # first index of a non-leaf node, from bottom to up
    for i in range(start, -1, -1): 
        heapify(a, n, i) 
    
    # sorting
    for end in range(n - 1, 0, -1): 
        swap(a, 0, end)

        # heapify the root element with a tree smaller than the previous one
        heapify(a, end, 0) 



def mergeSort(a):
    
    if len(a) < 2: # exit condition

        return        
    
    else:    

        m = len(a) // 2 # middle index
        l, r = a[:m], a[m:] # left and right halves

        mergeSort(l)
        mergeSort(r)

        i = j = k = 0

        while i < len(l) and j < len(r):
            if l[i] < r[j]:
                a[k] = l[i]
                i += 1
            else:
                a[k] = r[j]
                j += 1
            k += 1

        while i < len(l):
            a[k]=l[i]
            i += 1
            k += 1

        while j < len(r):
            a[k]=r[j]
            j += 1
            k += 1


def quickSort(a, l=0, h=None, f=lomutosPartition): 
    if h is None:
        h = len(a) -1

    if l < h:
        i = f(a, l, h)
        quickSort(a, l, i - 1)
        quickSort(a, i + 1, h)

    return a


def quickSortWithLomutosPartition(a):
    quickSort(a)


def quickSortWithHoaresPartition(a):
    quickSort(a, f=hoaresPartition)


def getMedianWithSelectionSort(a):
    return selectionSort(a)[(len(a) - 1) // 2]  # '//' floor division op


# Brute Force, O(n log n)
def getMedianWithPythonBuiltinSort(a):
    a.sort()
    return a[(len(a) - 1) // 2]    


def plotResults(data):
    # defines font size and line width
    sns.set(font_scale=1, rc={"lines.linewidth": 2})

    # defines plot size
    plt.rcParams["figure.figsize"] = [20, 10]

    grid = sns.lineplot(data=data, x="Size of Random Inputs", y="CPU Time in Seconds", hue="Algorithm", style="Algorithm")
    
    grid.set(yscale="log")

    plt.show()  


def populateArray(size):
    a = [None] * size
    for i in range(0, size): 
        a[i] = random.rand() % 50000
    return a


def execFunction(f, params, runs=1, randomState=None, debug=False):
    t = 0
    for i in range(runs):
        s = params['listSize']

        if not randomState:
            random.seed(i * s)
        else:
            random.seed(randomState)    

        a = populateArray(s) 
        
        start = timeit.default_timer()
        f(a)
        end = timeit.default_timer()
        t += end - start

        if debug:
            test = a[len(a) // 2]
            control = sorted(a)[len(a) // 2]
            assert control == test

    return f.__name__, t / runs


def run(mode: str = "parallel"):

    # defines the algorithms to be processed
    algorithms = [bubbleSort, insertionSortWithSentinel, heapSort, mergeSort, quickSortWithLomutosPartition]
    algorithmsLabels = {
        bubbleSort.__name__: "Bubble Sort",
        insertionSortWithSentinel.__name__: "Insertion Sort",
        heapSort.__name__: "Heap Sort",
        mergeSort.__name__: "Merge Sort", 
        quickSortWithLomutosPartition.__name__: "Quick Sort"
    }

    results = {}

    #defines the number of executionsNumber each algorithm will be processed to find the average time
    executionsNumber = 3
 
    sizes = []
    for i in range(1, 6, 1):
        sizes.append(2 ** i * 100)

    print(sizes)

    for i in range(len(sizes)):
        print(f"Starting test {i+1}, with {sizes[i]} data items")

        params = {"listSize" : sizes[i]}
  
        # algorithms are now run in parallel to maximize CPU usage
        if mode == "parallel":
            r = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
               delayed(execFunction)(algorithm, params, runs=executionsNumber, randomState=i)
               for algorithm in algorithms
            )
            a, t = zip(*r)

            for j in range(len(a)):
                algLabel = algorithmsLabels[a[j]]
                results[f"{algLabel}:{sizes[i]}"] = t[j]

        elif mode == "serial":
            for alg in algorithms:
                algLabel = algorithmsLabels[alg.__name__]
                results[f"{algLabel}:{sizes[i]}"] = execFunction(alg, params, runs=executionsNumber, randomState=i)[1]
        else:
            raise ValueError("Invalid mode: "+str(
                mode))       

    print(f"Tests completed")

    df = pd.DataFrame.from_dict(results, orient="index",columns=["CPU Time in Seconds"])
    df["Algorithm"] = [i.split(":")[0] for i in df.index]
    df["Size of Random Inputs"] = [int(i.split(":")[1]) for i in df.index]

    plotResults(df)


if __name__ == "__main__":
    run(mode="serial")
