def binarySearch(L, k):
    l = 0
    r = len(L) - 1
    while (l <= r):
        m = (l + r) // 2 # floor division
        if (L[m] == k):
            return m
        elif (L[m] > k):
            r = m - 1
        else:
            l = m + 1
    return -1

def swap(L, i1, i2):
    L[i1], L[i2] = L[i2], L[i1]

def partition(L, l, h):

    if l != h:
        # choose random pivot
        pIdx = random.randint(l, h) 

        # moves pivot to end of the list
        L[h], L[pIdx] = L[pIdx], L[h]

    # select the last element to be the pivot
    p = L[h] 

    s = l - 1 
    for j in range(l, h):
        if L[j] <= p:
            s +=  1        
            L[s], L[j] = L[j], L[s] # swap
            
    L[s + 1], L[h] = L[h], L[s + 1] # swap
    return s + 1


def quickSelect(L, l, h, k):
    i = k - 1
    s = partition(L, l, h) # split point
    if (i == s):
        return L[i]
    else:
        return quickSelect(L, l, s - 1, k) if (s > i) else quickSelect(L, s + 1, h, k)
