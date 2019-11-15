import matplotlib.pyplot as plt
import numpy as np


def plot_2d_features(features, cluster_assignments=None, scales=None, i=1):
    x = [datapoint[0] for datapoint in features]
    y = [datapoint[1] for datapoint in features]

    plt.clf()
    plt.scatter(x, y, c=cluster_assignments, alpha=0.5)
    plt.title("i = {0}".format(i))

    if scales is not None:
        plt.ylim([scales["min_y"], scales["max_y"]])
        plt.xlim([scales["min_x"], scales["max_x"]])
        uniques, counts = np.unique(cluster_assignments, return_counts=True)
        text = ['C'+str(unique)+': '+str(counts) for unique, counts in zip(uniques, counts)]
        plt.text(scales["min_x"]+0.01, scales["max_y"]-0.01, '\n'.join(text), fontsize=14, verticalalignment='top')

    plt.savefig('visualisation/plot{0}.png'.format(i))
    plt.clf()


def get_plot_scales(latent_features):
    return {
        "min_x": min([latent_feature[0] for latent_feature in latent_features]) - 0.2,
        "max_x": max([latent_feature[0] for latent_feature in latent_features]) + 0.2,
        "min_y": min([latent_feature[1] for latent_feature in latent_features]) - 0.2,
        "max_y": max([latent_feature[1] for latent_feature in latent_features]) + 0.2
    }
