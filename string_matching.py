import time
import string
import random
# import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# defines the number of times each algorithm will be processed to obtain the
# average time
num_rounds = 20

SEQUENCE_LENGTH = 0.1
DR = dict()


def Naive(T, S):
    i, j = 0, 0
    while i < len(T):
        if len(T) - i == len(S):
            return T[i:] == S
        elif T[i:i + len(S)] == S:
            return True
        else:
            i += 1
            j = 0
    return False


def create_kmp_table(S):
    i, j = 0, 1
    T = [0] * len(S)
    while j < len(S):
        if S[i] == S[j]:
            T[j] = i + 1
            i += 1
            j += 1
        elif i == 0:
            T[j] = 0
            j += 1
        else:
            i = T[i - 1]
    return T


def KMP(T, S):
    if len(S) > len(T):
        return False
    table = create_kmp_table(S)
    i, j = 0, 0
    while i < len(T):
        if T[i] == S[j]:
            if j == len(S) - 1:
                return True
            i += 1
            j += 1
        else:
            if j > 0:
                j = table[j - 1]
            else:
                i += 1
    return False


def BMH(T, S):
    if len(S) > len(T):
        return False
    table = {}
    for i in range(len(S)):
        if i < len(S) - 1:
            table[S[i]] = len(S) - i - 1
        elif S[i] not in table:
            table[S[i]] = len(S)
    i = len(S) - 1
    while i < len(T):
        if T[i] != S[-1] or T[i-len(S)+1:i+1] != S:
            i += table[T[i]] if T[i] in table else len(S)
        else:
            return True
    return False

def BMH_2(text, pattern):
    m = len(pattern)
    n = len(text)
    if m > n:
        return -1
    skip = []
    for k in range(256):
        skip.append(m)
    for k in range(m - 1):
        skip[ord(pattern[k])] = m - k - 1
    skip = tuple(skip)
    k = m - 1
    while k < n:
        j = m - 1
        i = k
        while j >= 0 and text[i] == pattern[j]:
            j -= 1
            i -= 1
        if j == -1:
            return True
        k += skip[ord(text[k])]
    return False


def plot(data):
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Time'])
    df['Algorithm'] = [i.split("##")[0] for i in df.index]
    df['Size'] = [int(i.split("##")[1]) for i in df.index]

    # Defines font size and line width
    sns.set(font_scale=1, style="ticks", rc={"lines.linewidth": 2})

    # Defines plot size
    plt.rcParams['figure.figsize'] = [20, 10]

    chart = sns.lineplot(x='Size', y='Time', hue='Algorithm', data=df)

    # plt.yscale('log')
    chart.set(xticks=[i for i in df.Size])
    plt.show()


# calculates the executions average time
def avgTime(func, size, debug=True):
    t = 0
    for i in range(num_rounds):
        random.seed(size + i)
        T = "".join([random.choice(string.ascii_letters) for i in range(size)])
        start = random.randint(1, size)
        end = min(start + int(size * SEQUENCE_LENGTH + 1), size)
        S = T[start:end]

        start = time.time()
        p = func(T, S)
        end = time.time()
        t += end - start

        if debug:
            # add the result or check if it is the same
            if (size, i) not in DR:
                DR[(size, i)] = (p, T, S)
            else:
                (sp, sT, sS) = DR[(size, i)]
                if p != sp:
                    print(f"1. S={DR[(size, i)][2]}, found={DR[(size, i)][0]} \
                        and T={DR[(size, i)][1]}")
                    print(f"2. S={S}, found={p} and T={T} ")

                assert p == sp

    return t / num_rounds


def run():
    # defines the algorithms to be processed
    algorithms = [Naive, KMP, BMH]
    algorithms = [Naive, KMP, BMH]
    sizes = [10000, 20000, 30000, 40000, 50000]
    mapSizeToTime = dict()
    for i in range(len(sizes)):
        print(f"Starting collect {i + 1}")

        # map list size to algorithm average time
        for algorithm in algorithms:
            print('  > ', algorithm.__name__)
            mapSizeToTime[f"{algorithm.__name__}##{sizes[i]}"] = \
                avgTime(algorithm, sizes[i], True)
    print("Finish data collection")
    plot(mapSizeToTime)


if __name__ == "__main__":
    run()
