from sklearn.cluster import KMeans
from keras.layers import Layer, InputSpec
import keras.backend as K


def get_init_cluster_center(n_clusters, latent_features):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
    kmeans.fit_predict(latent_features)
    return kmeans.cluster_centers_


class ClusteringLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='cluster', shape=(self.n_clusters, input_dim), initializer='uniform')
        self.set_weights(self.initial_weights)

        super(ClusteringLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # sic! we cannot use numpy, list comprehension or build-in loops of python here due to tensorflow
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
