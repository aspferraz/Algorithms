import time
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

WEIGHT_DOMAIN = 100
PROFIT_DOMAIN = 1000
CAPACITY_PROBABILITY = 0.3

# defines the number of times each algorithm will be processed to obtain the
# average time
num_rounds = 5

# alg_results = dict()


def knapSackNaive(W, P, c):
    def ks(W, P, c, i):
        if i == 0 or c == 0:
            return 0
        if W[i-1] > c:
            return ks(W, P, c, i-1)
        else:
            return max(P[i-1] + ks(W, P, c - W[i-1], i-1), ks(W, P, c, i-1))

    return ks(W, P, c, len(W))


def knapSackMem(W, P, c):
    r = {}

    def ks(W, P, c, i):
        if i == 0 or c == 0:
            return 0
        if (c, i-1) in r:
            return r[(c, i-1)]
        if W[i-1] > c:
            r[(c, i-1)] = ks(W, P, c, i-1)
        else:
            r[(c, i-1)] = \
                max(P[i-1] + ks(W, P, c - W[i-1], i-1), ks(W, P, c, i-1))
        return r[(c, i-1)]

    return ks(W, P, c, len(W))


def knapSacTab(W, P, c):
    W.insert(0, 0)
    P.insert(0, 0)

    T_columns = c + 1
    T_rows = len(P)
    T = [[None for y in range(T_columns)] for x in range(T_rows)]

    for i in range(T_rows):
        for j in range(T_columns):
            if i == 0 or j == 0:
                T[i][j] = 0
            elif W[i] <= j:
                T[i][j] = max(T[i-1][j], P[i] + T[i-1][j - W[i]])
            else:
                T[i][j] = T[i-1][j]

    return T[-1][-1]


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
        random.seed(size+i)
        W = list(random.randint(WEIGHT_DOMAIN, size=size))
        P = list(random.randint(PROFIT_DOMAIN, size=size))
        c = random.randint(int(CAPACITY_PROBABILITY*size)*WEIGHT_DOMAIN)
        start = time.time()
        p = func(W, P, c)
        end = time.time()
        t += end - start

        if debug:
            # create a variable to store the debug results
            if 'DR' not in globals():
                global DR
                DR = dict()

            # add the result or check if it is the same
            if (size, i) not in DR:
                DR[(size, i)] = p
            else:
                assert p == DR[(size, i)]

    return t / num_rounds


def run():
    # defines the algorithms to be processed
    algorithms = [knapSackMem, knapSacTab]
    # algorithms = [knapSacTab]

    sizes = [5, 10, 15, 20, 25]
    # sizes = [100, 200, 300, 400, 500]

    mapSizeToTime = dict()
    for i in range(len(sizes)):
        print(f"Starting collect {i+1}")

        # map list size to algorithm average time
        for algorithm in algorithms:
            print('  > ', algorithm.__name__)
            mapSizeToTime[f"{algorithm.__name__ }##{sizes[i]}"] =  \
                avgTime(algorithm, sizes[i], True)

    print("Finish data collection")

    plot(mapSizeToTime)


if __name__ == "__main__":
    run()
