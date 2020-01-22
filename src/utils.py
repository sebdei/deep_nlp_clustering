import matplotlib.pyplot as plt


def plot_test_sets(test_1, test_2, test_3):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 12})

    plt.scatter(x=range(0, len(test_1[0]), 1), y=test_1[0], alpha=1, c="red", label=test_1[1])
    plt.plot(range(0, len(test_1[0]), 1), test_1[0], c="red")

    plt.scatter(x=range(0, len(test_2[0]), 1), y=test_2[0], alpha=1, c="green", label=test_2[1], marker="^")
    plt.plot(range(0, len(test_2[0]), 1), test_2[0], c="green")

    plt.scatter(x=range(0, len(test_3[0]), 1), y=test_3[0], alpha=1, c="blue", label=test_3[1], marker="s")
    plt.plot(range(0, len(test_3[0]), 1), test_3[0], c="blue")

    plt.ylim([0, 1])
    plt.xticks(range(0, len(test_1[0]), 2))

    plt.xlabel("Update interval")
    plt.ylabel("Fowlkes Mallows Index")

    plt.legend()
    plt.savefig('plot.png')
    plt.clf()
    