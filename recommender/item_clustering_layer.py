import tensorflow as tf
from sklearn.cluster import KMeans

layers = tf.contrib.layers
rnn = tf.contrib.rnn


class DeepTemporalClustering:
    """docstring for DeepTemporalClustering"""

    def __init__(self, params):

        self.n_clusters = params['n_clusters']
        self.alpha = params['alpha']

        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.z = params['embedding']
        z_shape = self.z.shape
        # clustering center
        # self.mu = tf.Variable(tf.zeros(shape=[self.n_clusters, z_shape[1], z_shape[2]]), name='mu')
        self.mu = tf.Variable(tf.zeros(shape=[self.n_clusters, z_shape[1]]), name='mu')

        with tf.name_scope('distribution'):
            # p, q: [N, seq_len, n_clusters]
            self.q = self.soft_assignment(self.z, self.mu)
            self.p = tf.placeholder(tf.float32, shape=[None, self.n_clusters])

            self.pred = tf.argmax(self.q, axis=1)

        with tf.name_scope('dtc-train'):
            # self.loss = self._kl_divergence(self.p, self.q)
            self.loss_kl = self._kl_divergence(self.p, self.q)
            self.loss = self.loss_kl
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            - embeddings: (N, L_tmp, dim)
            - cluster_centers: (n_clusters, L_tmp, dim)

        Return:
            - q_ij: (N, n_clusters)
        """
        # padding to [n_clusters, seq_length, dim]
        # cluster_padding_centers = tf.tile(cluster_centers, [1, self.z.shape[1]])
        # cluster_padding_centers = tf.reshape(cluster_padding_centers, shape=[-1, self.z.shape[1], self.z.shape[2]])
        expand_embeddings = tf.expand_dims(embeddings, axis=1)
        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(expand_embeddings - cluster_centers), axis=2) / self.alpha))

        q **= (self.alpha + 1.0) / 2.0
        # q : [N, n_clusters, seq_len]
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        # q : [N, seq_len, n_clusters]
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def tensor_target_distribution(self, q):
        p = q ** 2 / tf.reduce_sum(q, axis=0)
        p = p / tf.reduce_sum(p, axis=1, keepdims=True)
        return p

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))

    def get_assign_cluster_centers_op(self, features):
        # init mu
        print('Start training KMeans')
        kmeans = self.kmeans.fit(features.reshape(len(features), -1))
        print('Finish training KMeans')
        return tf.assign(self.mu,
                         kmeans.cluster_centers_.reshape(self.n_clusters, features.shape[1]))