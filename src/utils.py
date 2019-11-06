import matplotlib.pyplot as plt


def print_features(features, colors=None, cluster_assignments=None, scales=None, i=1):
    x = [datapoint[0] for datapoint in features]
    y = [datapoint[1] for datapoint in features]
    plt.clf()
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.title("i = {0}".format(i))
    if scales is not None:
        plt.ylim([scales["min_y"], scales["max_y"]])
        plt.xlim([scales["min_x"], scales["max_x"]])
        uniques, counts = np.unique(cluster_assignments, return_counts=True)
        text = ['C'+str(unique)+': '+str(counts) for unique, counts in zip(uniques, counts)]
        plt.text(scales["min_x"]+0.01, scales["max_y"]-0.01, '\n'.join(text), fontsize=14, verticalalignment='top')
    plt.savefig('visualisation/plot{0}.png'.format(i))
    plt.clf()
