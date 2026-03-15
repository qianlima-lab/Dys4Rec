import numpy as np
import tensorflow as tf
import random
from recommender.BasicRcommender_soft import BasicRecommender_soft
import time

import tensorflow.contrib.layers as layers
from recommender.item_clustering_layer import DeepTemporalClustering


class GRU4RecRecommender(BasicRecommender_soft):

    def __init__(self, dataModel, config):
        "继承后的重写语句"
        super(GRU4RecRecommender, self).__init__(dataModel, config)
        self.name = 'GRU4Rec'

        self.numFactor = config['numFactor']
        self.factor_lambda = config['factor_lambda']
        self.seq_length = config['seq_length']
        self.dropout_keep = config['dropout_keep']
        self.dropout_item = config['dropout_item']
        self.dropout_context1 = config['dropout_context1']
        self.dropout_context2 = config['dropout_context2']
        self.drop_memory = config['drop_memory']
        self.drop_user = config['dropout_user']

        self.rnn_unit_num = config['rnn_unit_num']
        self.rnn_layer_num = config['rnn_layer_num']
        self.rnn_cell = config['rnn_cell']
        self.decrease_train = False
        self.familiar_user_num = dataModel.familiar_user_num

        self.seq_direc = config['seq_direc']
        if self.seq_direc == 'ver':
            dataModel.generate_sequences_rnn_ver(self.seq_length)
        elif self.decrease_train:
            dataModel.generate_sequences_hor(self.seq_length, 1)
        else:
            dataModel.generate_sequences_rnn_hor(self.seq_length)

        # self.train_users = dataModel.train_users
        # self.train_sequences_input = dataModel.train_sequences_input
        # self.train_sequences_user_input = dataModel.train_sequences_user_input
        # self.train_sequences_target = dataModel.train_sequences_target
        # self.user_pred_sequences = dataModel.user_pred_sequences
        # self.user_pred_user_sequences = dataModel.user_pred_user_sequences
        #
        # self.trainSize = len(self.train_sequences_input)
        # self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1
        self.train_users = dataModel.train_users
        self.train_sequences_input = dataModel.train_sequences_input
        self.train_sequences_user_input = dataModel.train_sequences_user_input
        self.train_sequences_target = dataModel.train_sequences_target
        self.user_pred_sequences = dataModel.user_pred_sequences
        # self.user_pred_sequences_actions = dataModel.user_pred_sequences_actions

        self.user_pred_user_sequences = dataModel.user_pred_user_sequences
        # self.train_sequences_actions_input = dataModel.train_sequences_actions_input
        # self.train_sequences_actions_target = dataModel.train_sequences_actions_target

        # "每个 item 的得分"
        # self.input_seq_rating = dataModel.input_seq_rating
        # self.target_seq_rating = dataModel.target_seq_rating

        self.trainSize = len(self.train_sequences_input)
        self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1

        self.action_nums = 4
        self.n_clusters = 20

        # placeholders
        self.input_seq = tf.placeholder(tf.int32, [None, self.seq_length])
        self.input_user_id = tf.placeholder(tf.int32, [None, 1])
        self.test_input_seq = tf.placeholder(tf.int32, [None, self.seq_length])
        self.target_seq_pos = tf.placeholder(tf.int32, [None, self.seq_length])
        self.target_seq_neg = tf.placeholder(tf.int32, [None, self.neg_num * self.seq_length])
        self.pred_seq = tf.placeholder(tf.int32, [None, self.eval_item_num])
        self.dropout_keep_placeholder = tf.placeholder_with_default(1.0, shape=())

        # user/item embedding
        self.itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 0.1))
        self.loss_type = config['loss_type']
        self.target_weight = config['target_weight']
        self.center_embedding_weight = tf.Variable(tf.constant(0.5, shape=[1, self.numFactor]))

        self.center_feature_weight = 0.3
        """self.rnn_network = RNN_Compoment(
            rnn_unit_num=self.rnn_unit_num,
            rnn_layer_num=self.rnn_layer_num,
            rnn_cell=self.rnn_cell,
            output_size=self.numFactor,
            wordvec_size=self.numFactor,
            input_placeholder=None,
            max_review_length=self.seq_length,
            word_matrix=self.itemEmbedding,
            review_wordId_print=None,
            review_input_print=None,
            rnn_lambda=None,
            dropout_keep_prob=self.dropout_keep_placeholder,
            component_raw_output=None,
            item_pad_num=None
        ).return_network()"""
        if self.loss_type == 'soft':
            self.numK = config['numK']
        else:
            self.numK = 1

        self.prior_weight = tf.get_variable("priorweight", shape=[self.numK, self.rnn_unit_num],
                                            initializer=tf.random_uniform_initializer(
                                                minval=-1 / tf.sqrt(float(self.numFactor)),
                                                maxval=1 / tf.sqrt(float(self.numFactor))),
                                            dtype=tf.float32, trainable=True)
        "相当于item embedding matrix"
        self.cell = tf.contrib.rnn.GRUCell(self.rnn_unit_num)

        labels_vector1 = tf.constant(1.0, shape=[self.trainBatchSize * self.seq_length, self.numK, 1])
        labels_vector2 = tf.constant(0.0, shape=[self.trainBatchSize * self.seq_length, self.numK, self.neg_num])
        self.labels2 = tf.concat([labels_vector1, labels_vector2], axis=2)

        self.output_fc_W = tf.get_variable(
            name="output_fc_W",
            dtype=tf.float32,
            shape=[self.numK * self.numFactor, self.rnn_unit_num],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        self.output_item_embedding = tf.get_variable(
            name="output_item_embedding",
            dtype=tf.float32,
            shape=[self.numItem, self.numFactor],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        self.output_fc_b = tf.get_variable(
            name="output_fc_b",
            dtype=tf.float32,
            initializer=tf.constant(0.1, shape=[self.numK * self.numFactor])
        )
        self.denselayer = tf.get_variable("denselayer",
                                          shape=[self.numK * self.numFactor, self.numFactor + self.numFactor],
                                          initializer=tf.random_uniform_initializer(
                                              minval=-1 / tf.sqrt(float(self.numFactor)),
                                              maxval=1 / tf.sqrt(float(self.numFactor))),
                                          dtype=tf.float32, trainable=True)
        self.denseBias = tf.Variable(tf.random_normal([self.numK * self.numFactor], 0, 0.1))


        self.clustering_layer = None
        self.p_value = None

        # self.proba_prediction = None

    # def pick_top_n(self, preds, vocab_size, top_n=5):
    #     p = np.squeeze(preds)
    #     # 将除了top_n个预测值的位置都置为0
    #     p[np.argsort(p)[:-top_n]] = 0
    #     # 归一化概率
    #     p = p / np.sum(p)
    #     # 随机选取一个 item
    #     c = np.random.choice(vocab_size, 1, p=p)[0]
    #     return c
    #
    # def sample(self, n_samples, prime, vocab_size):
    #     samples = [c for c in prime]
    #     new_state = None
    #     preds = np.ones((vocab_size,))  # for prime=[]
    #     for c in prime:
    #         x = np.zeros((1, 1))
    #         # 输入单个字符
    #         x[0, 0] = c
    #         feed = {self.inputs: x,
    #                 self.keep_prob: 1.,
    #                 self.initial_state: new_state}
    #         preds, new_state = sess.run([self.proba_prediction, self.final_state],
    #                                     feed_dict=feed)
    #
    #     c = pick_top_n(preds, vocab_size)
    #     # 添加字符到samples中
    #     samples.append(c)
    #
    #     # 不断生成字符，直到达到指定数目
    #     for i in range(n_samples):
    #         x = np.zeros((1, 1))
    #         x[0, 0] = c
    #         feed = {self.inputs: x,
    #                 self.keep_prob: 1.,
    #                 self.initial_state: new_state}
    #         preds, new_state = sess.run([self.proba_prediction, self.final_state],
    #                                     feed_dict=feed)
    #
    #         c = pick_top_n(preds, vocab_size)
    #         samples.append(c)
    #
    #     return np.array(samples)

    def Action_Fusion_Block(self, item_embedding, action_labels):

        OUTPUT_SIZE = self.numFactor
        # [batch size, seq_length, embedding_dims]
        finished_embedding = layers.linear(inputs=item_embedding, num_outputs=OUTPUT_SIZE)
        islike_embedding = layers.linear(inputs=finished_embedding, num_outputs=OUTPUT_SIZE)
        iscom_embedding = layers.linear(inputs=finished_embedding, num_outputs=OUTPUT_SIZE)
        isfollow_embedding = layers.linear(inputs=finished_embedding, num_outputs=OUTPUT_SIZE)

        # prediction layer
        prediction_layer = tf.get_variable(
            name="action_prediction_layer",
            dtype=tf.float32,
            shape=[self.numFactor, 1],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        prediction_layer_bias = tf.get_variable(
            name="action_prediction_layer_bias",
            dtype=tf.float32,
            shape=[1],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        # reshape the embedding [batch*seq_length, embedding_dims]
        reshape_finished_embedding = tf.reshape(finished_embedding, shape=[-1, self.numFactor])
        reshape_islike_embedding = tf.reshape(islike_embedding, shape=[-1, self.numFactor])
        reshape_iscom_embedding = tf.reshape(iscom_embedding, shape=[-1, self.numFactor])
        reshape_isfollow_embedding = tf.reshape(isfollow_embedding, shape=[-1, self.numFactor])

        # [batch size*seq_length, 1]
        finished_predict = tf.nn.sigmoid(
            tf.matmul(reshape_finished_embedding, prediction_layer) + prediction_layer_bias)
        islike_predict = tf.nn.sigmoid(tf.matmul(reshape_islike_embedding, prediction_layer) + prediction_layer_bias)
        iscom_predict = tf.nn.sigmoid(tf.matmul(reshape_iscom_embedding, prediction_layer) + prediction_layer_bias)
        isfollow_predict = tf.nn.sigmoid(
            tf.matmul(reshape_isfollow_embedding, prediction_layer) + prediction_layer_bias)

        # [batch size, seq_length, 1]

        # [batch size*seq_length, action_nums]
        total_prediction = tf.concat([islike_predict, iscom_predict, isfollow_predict, finished_predict], axis=1)
        action_mse_loss = tf.losses.mean_squared_error(labels=tf.reshape(action_labels, shape=[-1, self.action_nums]),
                                                       predictions=total_prediction)

        # according prediction labels, as the feature weight, to fusion the embedding
        soft_prediction = tf.nn.softmax(total_prediction, axis=1)
        # reshape [batch size*seq_length, action_nums, 1]
        soft_prediction = tf.reshape(soft_prediction, shape=[-1, self.action_nums, 1])
        # [batch size*seq_length, embedding_dims, action_nums]
        total_embedding = tf.stack([reshape_islike_embedding, reshape_iscom_embedding,
                                    reshape_isfollow_embedding, reshape_finished_embedding], axis=2)
        # [batch size*seq_length, embedding_dims, 1]
        fusion_embedding = tf.matmul(total_embedding, soft_prediction)
        # reshape [batch size, seq_length, embedding_dims]
        fusion_embedding = tf.reshape(fusion_embedding, shape=[-1, self.seq_length, self.numFactor])

        return fusion_embedding, action_mse_loss

    def buildModel(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:

            item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.input_seq)
            # Batch size x time steps x features.
            userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.input_user_id),
                                       [-1, self.numFactor])
            user_embedding_drop = tf.nn.dropout(userEmbedding, self.drop_user)

            # padding to shape [batch, input_seq, numFactor] in order to fusion the input item seq embedding
            user_padding_embedding = tf.tile(user_embedding_drop, [1, self.seq_length])
            user_padding_embedding = tf.reshape(user_padding_embedding, shape=[-1, self.seq_length, self.numFactor])

            item_embed_input = tf.nn.dropout(tf.reshape(item_embed_input, [-1, self.seq_length, self.numFactor]),
                                             self.dropout_item)

            test_item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.test_input_seq)
            # Batch size x time steps x features.
            test_item_embed_input = tf.reshape(test_item_embed_input, [-1, self.seq_length, self.numFactor])

            clustering_params = {
                'n_clusters': self.n_clusters,
                'alpha': 1,
                'embedding': tf.reshape(item_embed_input, shape=[-1, self.numFactor])
            }

            self.clustering_layer = DeepTemporalClustering(params=clustering_params)
            center_embedding = tf.nn.embedding_lookup(self.clustering_layer.mu, self.clustering_layer.pred)
            # center_embedding = tf.reshape(center_embedding, shape=[-1, self.seq_length, self.numFactor])
            # weight add the center [batch, embedding_dims]
            # center_fusion_embedding = tf.matmul(self.clustering_layer.q, self.clustering_layer.mu)
            # center_fusion_embedding = tf.reshape(center_fusion_embedding, shape=[-1, self.seq_length, self.numFactor])

            fusion_embedding = tf.add(item_embed_input,
                                      tf.multiply(tf.clip_by_value(self.user_embedding_weight, 0.1, 1.0),
                                                  user_padding_embedding))

            item_embed_input = tf.nn.dropout(tf.reshape(fusion_embedding, [-1, self.seq_length, self.numFactor]),
                                             self.drop_memory)

            #concat_embedding = tf.concat([tf.reshape(item_embed_input, shape=[-1, self.numFactor]), center_embedding], axis=1)
            #fusion_embedding = tf.reshape(tf.tanh(tf.matmul(concat_embedding, self.denselayer, transpose_b=True)
            #                                        + self.denseBias), [-1, self.numK, self.numFactor])
            #fusion_embedding = tf.add(item_embed_input,
            #                           tf.multiply(tf.clip_by_value(self.center_embedding_weight, 0.1, 1.0),
            #                                      center_embedding))
            item_embed_input = tf.nn.dropout(item_embed_input, self.drop_memory)

            rnn_outputs, curr_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=item_embed_input,
                dtype=tf.float32,
            )

            split_outputs = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.rnn_unit_num]), self.dropout_context1)
            prior_weight = tf.reshape(tf.matmul(split_outputs, self.prior_weight, transpose_b=True),
                                      [-1, self.numK, 1])
            context_vector = tf.tanh(tf.matmul(split_outputs, self.output_fc_W, transpose_b=True)
                                                + self.output_fc_b)
            context_vector = tf.reshape(context_vector, shape=[-1, self.numK, self.numFactor])

            # [batch, embed_dims]
            context_drop = tf.nn.dropout(context_vector, self.dropout_context2)

            test_q = self.clustering_layer.soft_assignment(embeddings=tf.reshape(test_item_embed_input, shape=[-1, self.numFactor]),
                                                           cluster_centers=self.clustering_layer.mu)
            test_p = self.clustering_layer.tensor_target_distribution(test_q)
            test_pred = tf.argmax(test_q, axis=1)
            test_center_embedding = tf.nn.embedding_lookup(self.clustering_layer.mu, test_pred)
            test_concat_embedding = tf.concat([tf.reshape(test_item_embed_input, shape=[-1, self.numFactor]), test_center_embedding], axis=1)
            test_fusion_embedding = tf.reshape(tf.tanh(tf.matmul(test_concat_embedding, self.denselayer, transpose_b=True)
                                                  + self.denseBias), [-1, self.numK, self.numFactor])
            test_item_embed_input = tf.nn.dropout(test_fusion_embedding, self.drop_memory)

            test_rnn_outputs, test_curr_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=test_item_embed_input,
                dtype=tf.float32,
            )

            test_split_outputs = tf.reshape(test_rnn_outputs, [-1, self.seq_length, self.rnn_unit_num])
            test_gru_vector = tf.reshape(test_split_outputs[:, -1:, :], [-1, self.rnn_unit_num])
            test_prior_weight = tf.reshape(tf.matmul(test_gru_vector, self.prior_weight, transpose_b=True),
                                           [-1, self.numK, 1])
            test_context_vector = tf.tanh(tf.matmul(test_gru_vector, self.output_fc_W, transpose_b=True)
                                                     + self.output_fc_b)
            test_context_vector = tf.reshape(test_context_vector, shape=[-1, self.numK, self.numFactor])
            pos_embeds = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, self.target_seq_pos),
                                    [-1, 1, self.numFactor])
            neg_embeds = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, self.target_seq_neg),
                                    [-1, self.neg_num, self.numFactor])

            element_pos = tf.matmul(context_drop, pos_embeds, transpose_b=True)
            element_neg = tf.matmul(context_drop, neg_embeds, transpose_b=True)

            if self.loss_type == 'bpr':
                self.cost = self.get_bpr_pred(element_pos, element_neg) + self.clustering_layer.loss_kl
            else:
                self.cost = self.get_soft_pred(prior_weight, element_pos, element_neg)
            "训练图和测试图往往不一样，在建立图的时候考虑清楚， 一般先定义好图，然后在需要的时候再运行，运行的时候再定义session"
            "测试时相当于从头开始走测试图，测试的数据也是从头开始"

            self.r_pred = self.test_pred(test_prior_weight, test_context_vector, self.pred_seq)
            # self.proba_prediction = tf.matmul(x, output_fc_W) + softmax_b

    def get_bpr_pred(self, element_pos, element_neg):
        bpr_loss = - tf.reduce_sum(tf.reduce_mean(tf.log(tf.sigmoid(-(element_neg - element_pos)) + 1e-7), axis=2))
        return bpr_loss

    def get_soft_pred(self, prior_weight, element_pos, element_neg):
        sig_weight = tf.nn.sigmoid(prior_weight)
        element_wise_mul = tf.nn.softmax(tf.concat([element_pos, element_neg], axis=2), axis=2)
        mse_log = tf.abs(element_wise_mul - sig_weight)
        mse_t = (self.target_weight + element_wise_mul) - self.target_weight * (mse_log + element_wise_mul)
        mse_p = tf.log(mse_t + 1e-7)
        mse_n = tf.log((1 - mse_t) + 1e-7)
        mse_loss = tf.reduce_mean(tf.reshape(tf.reduce_mean(tf.reduce_sum(self.labels2 * (mse_n - mse_p) - mse_n,
                                                                          axis=2), axis=1), [-1, 1]))
        return mse_loss

    def test_pred(self, test_prior_weight, test_context_vector, test_item_ids):
        test_item_embedding = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, test_item_ids),
                                         [-1, self.eval_item_num, self.numFactor])
        if self.numK > 1:
            pred_soft = tf.nn.softmax(tf.reshape(tf.matmul(test_context_vector, test_item_embedding, transpose_b=True),
                                                 [-1, self.numK, self.eval_item_num]), axis=2)
            pred_dot = tf.reshape(tf.reduce_sum(tf.multiply(pred_soft, test_prior_weight), axis=1),
                                  [-1, self.eval_item_num])
        else:
            pred_dot = tf.reshape(tf.matmul(test_context_vector, test_item_embedding, transpose_b=True),
                                  [-1, self.eval_item_num])
        return pred_dot

    def trainEachBatch(self, epochId, batchId):
        totalLoss = 0
        start = time.time()
        input_seq_batch, pos_seq_batch, neg_seq_batch = self.getTrainData(batchId)

        # print(batchId)
        if epochId == 0 and batchId == 0:
            # init mu
            item_embedding_val = self.sess.run(self.itemEmbedding)
            # using item embedding center
            assign_mu_op = self.clustering_layer.get_assign_cluster_centers_op(item_embedding_val)
            _ = self.sess.run(assign_mu_op)

        if epochId % 10 == 0:
            q = self.sess.run(self.clustering_layer.q, feed_dict={
                self.input_seq: input_seq_batch,
                self.dropout_keep_placeholder: self.dropout_keep
            })

            p = self.clustering_layer.target_distribution(q)
            self.p_value = p
            if batchId == 0:
                print(q[0])
                print(p[0])
                print(self.sess.run(self.clustering_layer.mu))

        self.optimizer.run(feed_dict={
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep,
            self.clustering_layer.p: self.p_value
        })

        loss = self.cost.eval(feed_dict={
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep,
            self.clustering_layer.p: self.p_value
        })

        kl_loss = self.clustering_layer.loss_kl.eval(feed_dict={
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep,
            self.clustering_layer.p: self.p_value
        })
        # if batchId == 0:
        #     print(loss)
        #     print(kl_loss)

        totalLoss += loss
        end = time.time()
        if epochId % 5 == 0 and batchId == 0:
            self.logger.info("----------------------------------------------------------------------")
            self.logger.info(
                "batchId: %d epoch %d/%d   batch_loss: %.4f   time of a batch: %.4f" % (
                    batchId, epochId, self.maxIter, totalLoss, (end - start)))

            self.evaluateRanking(epochId, batchId)
        return totalLoss, None

    def getTrainData(self, batchId):
        # compute start and end
        start = time.time()

        user_batch = []
        input_seq_batch = []
        pos_seq_batch = []
        neg_seq_batch = []

        start_idx = batchId * self.trainBatchSize
        end_idx = start_idx + self.trainBatchSize

        if end_idx > self.trainSize:
            end_idx = self.trainSize
            start_idx = end_idx - self.trainBatchSize

        if end_idx == start_idx:
            start_idx = 0
            end_idx = start_idx + self.trainBatchSize

        user_batch = self.train_users[start_idx:end_idx]
        input_seq_batch = self.train_sequences_input[start_idx:end_idx]
        pos_seq_batch = self.train_sequences_target[start_idx:end_idx]

        if self.seq_direc == 'ver':
            neg_start_index = end_idx
            if neg_start_index + (end_idx - start_idx) > self.trainSize:
                neg_start_index = 0
            neg_end_index = neg_start_index + (end_idx - start_idx)
            neg_seq_batch = self.train_sequences_target[neg_start_index:neg_end_index]
        else:
            for Idx in range(len(user_batch)):
                neg_items = []
                positiveItems = pos_seq_batch[Idx]
                for i in range(self.seq_length * self.neg_num):
                    negativeItemIdx = random.randint(0, self.numItem - 1)
                    while negativeItemIdx in positiveItems:
                        negativeItemIdx = random.randint(0, self.numItem - 1)
                    neg_items.append(negativeItemIdx)
                neg_seq_batch.append(neg_items)

        input_seq_batch = np.array(input_seq_batch)
        pos_seq_batch = np.array(pos_seq_batch)
        neg_seq_batch = np.array(neg_seq_batch)

        end = time.time()
        # self.logger.info("time of collect a batch of data: " + str((end - start)) + " seconds")
        # self.logger.info("batch Id: " + str(batchId))

        return input_seq_batch, pos_seq_batch, neg_seq_batch

    def getPredList_ByUserIdxList(self, user_idices):
        end0 = time.time()
        # build test batch
        input_seq = []
        target_seq = []

        for userIdx in user_idices:
            input_seq.append(self.user_pred_sequences[userIdx])
            target_seq.append(self.evalItemsForEachUser[userIdx])

        input_seq = np.array(input_seq)
        target_seq = np.array(target_seq)

        end1 = time.time()

        # q = self.sess.run(self.clustering_layer.q, feed_dict={
        #     self.input_seq: input_seq,
        #     self.dropout_keep_placeholder: 1,
        # })
        # p = self.clustering_layer.target_distribution(q)

        predList = self.sess.run(self.r_pred, feed_dict={
            self.test_input_seq: input_seq,
            self.pred_seq: target_seq,
        })
        end2 = time.time()

        output_lists = []
        for i in range(len(user_idices)):
            recommendList = {}
            start = i * self.eval_item_num
            end = start + self.eval_item_num
            for j in range(end - start):
                recommendList[target_seq[i][j]] = predList[i][j]
            sorted_RecItemList = sorted(recommendList, key=recommendList.__getitem__, reverse=True)[0:self.topN]
            output_lists.append(sorted_RecItemList)
        end3 = time.time()

        return output_lists, end1 - end0, end2 - end1, end3 - end2

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.input_seq)
            userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.input_user_id),
                                       [-1, self.numFactor])
            user_embedding_drop = tf.nn.dropout(userEmbedding, self.drop_user)
            # padding to shape [batch, input_seq, numFactor] in order to fusion the input item seq embedding
            user_padding_embedding = tf.tile(user_embedding_drop, [1, self.seq_length])
            user_padding_embedding = tf.reshape(user_padding_embedding, shape=[-1, self.seq_length, self.numFactor])

            item_embed_input = tf.nn.dropout(tf.concat([item_embed_input, user_padding_embedding], axis=1),
                                             self.drop_user)

            # Batch size x time steps x features.
            item_embed_input = tf.nn.dropout(tf.reshape(item_embed_input, [-1, self.seq_length, self.numFactor]),
                                             self.dropout_item)
            # test input
            test_item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.test_input_seq)
            'test seq uid '
            userEmbedding_test = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.u_id_test),
                                            [-1, self.numFactor])
            user_embedding_drop_test = tf.nn.dropout(userEmbedding_test, self.drop_user)
            # padding to shape [batch, input_seq, numFactor] in order to fusion the input item seq embedding
            user_padding_embedding_test = tf.tile(user_embedding_drop, [1, self.seq_length])
            user_padding_embedding_test = tf.reshape(user_padding_embedding_test,
                                                     shape=[-1, self.seq_length, self.numFactor])
            test_item_embed_input = tf.nn.dropout(
                tf.concat([test_item_embed_input, user_padding_embedding_test], axis=1),
                self.drop_user)

            clustering_params = {
                'n_clusters': self.n_clusters,
                'alpha': 1,
                'embedding': tf.reshape(item_embed_input, shape=[-1, self.numFactor])
            }
            self.clustering_layer = DeepTemporalClustering(params=clustering_params)

            center_embedding = tf.nn.embedding_lookup(self.clustering_layer.mu, self.clustering_layer.pred)

            # center_embedding = tf.matmul(self.clustering_layer.p, self.clustering_layer.mu)
            center_embedding = tf.reshape(center_embedding, shape=[-1, self.seq_length, self.numFactor])

            fusion_embedding = tf.add(item_embed_input,
                                      tf.multiply(tf.clip_by_value(self.center_embedding_weight, 0.1, 1.0),
                                                  center_embedding))
            item_embed_input = tf.nn.dropout(tf.reshape(fusion_embedding, [-1, self.seq_length, self.numFactor]),
                                             self.drop_memory)
            rnn_outputs, curr_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=item_embed_input,
                dtype=tf.float32,
                initial_state=self.initial_state
            )
            rnn_outputs, train_updated_states = self.split_rnn_outputs(rnn_outputs)

            test_q = self.clustering_layer.soft_assignment(
                embeddings=tf.reshape(test_item_embed_input, shape=[-1, self.numFactor]),
                cluster_centers=self.clustering_layer.mu)
            test_p = self.clustering_layer.tensor_target_distribution(test_q)
            test_pred = tf.argmax(test_p, axis=1)
            test_center_embedding = tf.nn.embedding_lookup(self.clustering_layer.mu, test_pred)
            # test_center_embedding = tf.matmul(test_p, self.clustering_layer.mu)
            test_fusion_embedding = tf.add(test_item_embed_input,
                                           tf.multiply(tf.clip_by_value(self.center_embedding_weight, 0.1, 1.0),
                                                       test_center_embedding))
            test_item_embed_input = tf.nn.dropout(
                tf.reshape(test_fusion_embedding, [-1, self.seq_length, self.numFactor]),
                self.drop_memory)

            # test_center_embedding = tf.reshape(test_center_embedding, shape=[-1, self.seq_length, self.numFactor])
            # test_item_embed_input = tf.add(test_item_embed_input,
            #                              tf.multiply(tf.clip_by_value(self.center_embedding_weight, 0.1, 1.0),
            #                                           test_center_embedding))
            test_rnn_outputs, test_curr_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=test_item_embed_input,
                dtype=tf.float32,
                initial_state=self.initial_state
            )
            test_rnn_outputs, test_updated_states = self.split_rnn_outputs(test_rnn_outputs)

            split_outputs = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.rnn_unit_num]), self.dropout_context1)
            split_outputs = tf.reshape(split_outputs, [-1, self.rnn_unit_num])
            '''
            user_padding_embedding = tf.reshape(user_padding_embedding, [-1, self.numFactor])
            # dot_embedding = tf.multiply(split_outputs, user_padding_embedding)
            gru_vector = tf.nn.dropout(tf.concat([split_outputs, user_padding_embedding], axis=1),
                                       self.drop_user)

            gru_vector = tf.reshape(split_outputs, [-1, self.rnn_unit_num + self.numFactor])
            user_embedding_new = tf.reshape(tf.tanh(tf.matmul(split_outputs, self.denselayer, transpose_b=True)
                                                    + self.denseBias), [-1, self.numK, self.numFactor])
            '''
            prior_weight = None

            context_vector = tf.reshape(tf.tanh(tf.matmul(split_outputs, self.output_fc_W, transpose_b=True)
                                                + self.output_fc_b), [-1, self.numK, self.numFactor])

            context_drop = tf.nn.dropout(context_vector, self.dropout_context2)

            # fusion the final hidden state and user embedding

            test_split_outputs = tf.reshape(test_rnn_outputs, [-1, self.seq_length, self.rnn_unit_num])
            test_gru_vector = tf.reshape(test_split_outputs[:, -1:, :], [-1, self.rnn_unit_num])

            # userEmbedding_test = tf.reshape(userEmbedding_test, [-1, self.numFactor])
            # test_dot_embedding = tf.multiply(test_gru_vector, userEmbedding_test)
            # test_gru_vector = tf.concat([test_gru_vector, userEmbedding_test], axis=1)

            # test_gru_vector = tf.reshape(test_gru_vector, [-1, self.rnn_unit_num + self.numFactor])
            # test_user_embedding_new = tf.reshape(tf.tanh(tf.matmul(test_gru_vector, self.denselayer, transpose_b=True)
            #                                             + self.denseBias), [-1, self.numK, self.numFactor])

            test_context_vector = tf.reshape(tf.tanh(tf.matmul(test_gru_vector, self.output_fc_W, transpose_b=True)
                                                     + self.output_fc_b), [-1, self.numK, self.numFactor])

            pos_embeds = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, self.target_seq_pos),
                                    [-1, 1, self.numFactor])
            neg_embeds = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, self.target_seq_neg),
                                    [-1, self.neg_num, self.numFactor])

            element_pos = tf.matmul(context_drop, pos_embeds, transpose_b=True)
            # element_pos = tf.multiply(element_pos,
            #                           tf.reshape(self.target_seq_rating_holder,
            #                                      shape=[self.trainBatchSize*self.seq_length, 1, 1]))
            element_neg = tf.matmul(context_drop, neg_embeds, transpose_b=True)

            if self.loss_type == 'bpr':
                self.cost = self.get_bpr_pred(element_pos, element_neg) + 2000 * self.clustering_layer.loss_kl
            else:
                self.cost = self.get_soft_pred(prior_weight, element_pos, element_neg)
            "训练图和测试图往往不一样，在建立图的时候考虑清楚， 一般先定义好图，然后在需要的时候再运行，运行的时候再定义session"
            "测试时相当于从头开始走测试图，测试的数据也是从头开始"

            test_prior_weight = None
            self.r_pred = self.test_pred(test_prior_weight, test_context_vector, self.pred_seq)
            # self.proba_prediction = tf.matmul(x, output_fc_W) + softmax_b

        '''
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.input_seq)

            userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.input_user_id),
                                       [-1, self.numFactor])
            user_embedding_drop = tf.nn.dropout(userEmbedding, self.drop_user)

            # padding to shape [batch, input_seq, numFactor] in order to fusion the input item seq embedding
            user_padding_embedding = tf.tile(user_embedding_drop, [1, self.seq_length])
            user_padding_embedding = tf.reshape(user_padding_embedding, shape=[-1, self.seq_length, self.numFactor])

            # Batch size x time steps x features.
            item_embed_input = tf.nn.dropout(tf.reshape(item_embed_input, [-1, self.seq_length, self.numFactor]),
                                             self.dropout_item)
            item_embed_input = tf.reshape(item_embed_input, [-1])

            clustering_params = {
                'n_clusters': self.n_clusters,
                'alpha': 1,
                'embedding': tf.reshape(item_embed_input, shape=[-1, self.numFactor])
            }

            self.clustering_layer = DeepTemporalClustering(params=clustering_params)
            item_center_embedding = tf.nn.embedding_lookup(self.clustering_layer.mu, self.clustering_layer.pred)
            item_center_embedding = tf.reshape(item_center_embedding, shape=[-1, self.seq_length, self.numFactor])
            item_center_embedding = tf.reduce_mean(item_center_embedding, axis=1)

            # test input
            test_item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.test_input_seq)
            'test seq uid '
            test_item_embed_input = tf.reshape(test_item_embed_input, shape=[-1, 1])
            userEmbedding_test = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.u_id_test),
                                            [-1, self.numFactor])
            user_padding_embedding_test = tf.tile(userEmbedding_test, [1, self.seq_length])
            user_padding_embedding_test = tf.reshape(user_padding_embedding_test, shape=[-1, self.seq_length, self.numFactor])

            test_q = self.clustering_layer.soft_assignment(
                embeddings=tf.reshape(test_item_embed_input, shape=[-1, self.numFactor]),
                cluster_centers=self.clustering_layer.mu)

            test_pred = tf.argmax(test_q, axis=1)
            test_center_embedding = tf.nn.embedding_lookup(self.clustering_layer.mu, test_pred)
            test_center_embedding = tf.reshape(test_center_embedding, shape=[-1, self.seq_length, self.numFactor])
            test_center_embedding = tf.reduce_mean(test_center_embedding, axis=1)

            # fusion user embedding and item embedding weighted adding: x+a*y+b*z a is the weight
            fusion_embedding = tf.add(item_embed_input,
                                      tf.multiply(tf.clip_by_value(self.user_embedding_weight, 0.1, 1.0),
                                                  user_padding_embedding))

            item_embed_input = tf.nn.dropout(tf.reshape(fusion_embedding, [-1, self.seq_length, self.numFactor]),
                                             self.drop_memory)


            # Batch size x time steps x features.
            test_fusion_embedding = tf.add(test_item_embed_input,
                                      tf.multiply(tf.clip_by_value(self.user_embedding_weight, 0.1, 1.0),
                                                  user_padding_embedding_test))

            test_item_embed_input = tf.reshape(test_fusion_embedding, [-1, self.seq_length, self.numFactor])

            rnn_outputs, curr_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=item_embed_input,
                dtype=tf.float32,
                initial_state=self.initial_state
            )
            rnn_outputs, train_updated_states = self.split_rnn_outputs(rnn_outputs)

            test_rnn_outputs, test_curr_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=test_item_embed_input,
                dtype=tf.float32,
                initial_state=self.initial_state
            )
            test_rnn_outputs, test_updated_states = self.split_rnn_outputs(test_rnn_outputs)

            split_outputs = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.rnn_unit_num]), self.dropout_context1)

            #split_outputs = tf.nn.dropout(tf.reshape(rnn_outputs[:, -1, :], [-1, self.rnn_unit_num]), self.dropout_context1)
            #gru_vector = tf.nn.dropout(tf.concat([split_outputs,
            #                                      tf.reshape(user_embedding_drop, [-1, self.numFactor]), item_center_embedding], axis=1),
            #                           self.drop_user)
            gru_vector = tf.reshape(split_outputs, [-1, self.rnn_unit_num + self.numFactor])
            user_embedding_new = tf.reshape(tf.tanh(tf.matmul(gru_vector, self.denselayer, transpose_b=True)
                                                    + self.denseBias), [-1, self.numK, self.numFactor])

            prior_weight = tf.reshape(tf.matmul(split_outputs, self.prior_weight, transpose_b=True),
                                      [-1, self.numK, 1])

            # context_vector = tf.reshape(tf.tanh(tf.matmul(split_outputs, self.output_fc_W, transpose_b=True)
            #                                     + self.output_fc_b), [-1, self.numK, self.numFactor])

            context_drop = tf.nn.dropout(user_embedding_new, self.dropout_context2)

            # fusion the final hidden state and user embedding

            test_split_outputs = tf.reshape(test_rnn_outputs, [-1, self.seq_length, self.rnn_unit_num])
            test_gru_vector = tf.reshape(test_split_outputs[:, -1:, :], [-1, self.rnn_unit_num])

            test_gru_vector = tf.concat([test_gru_vector, tf.reshape(userEmbedding_test, [-1, self.numFactor]), test_center_embedding], axis=1)

            # test_gru_vector = tf.reshape(test_gru_vector, [-1, self.rnn_unit_num + self.numFactor])
            test_user_embedding_new = tf.reshape(tf.tanh(tf.matmul(test_gru_vector, self.denselayer, transpose_b=True)
                                                    + self.denseBias), [-1, self.numK, self.numFactor])

            # test_prior_weight = tf.reshape(tf.matmul(test_user_embedding_new, self.prior_weight, transpose_b=True),
            #                                [-1, self.numK, 1])
            test_prior_weight = None
            # test_context_vector = tf.reshape(tf.tanh(tf.matmul(test_gru_vector, self.output_fc_W, transpose_b=True)
            #                                          + self.output_fc_b), [-1,  self.numK, self.numFactor])

            pos_embeds = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, self.target_seq_pos),
                                    [-1, 1, self.numFactor])
            neg_embeds = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, self.target_seq_neg),
                                    [-1, self.neg_num, self.numFactor])

            element_pos = tf.matmul(context_drop, pos_embeds, transpose_b=True)
            # element_pos = tf.multiply(element_pos,
            #                           tf.reshape(self.target_seq_rating_holder,
            #                                      shape=[self.trainBatchSize*self.seq_length, 1, 1]))
            element_neg = tf.matmul(context_drop, neg_embeds, transpose_b=True)

            budget_loss = self.compute_budget_loss(train_updated_states, self.cost)
            if self.loss_type == 'bpr':
                self.cost = self.get_bpr_pred(element_pos, element_neg) + self.clustering_layer.loss_kl
            else:
                self.cost = self.get_soft_pred(prior_weight, element_pos, element_neg)
            "训练图和测试图往往不一样，在建立图的时候考虑清楚， 一般先定义好图，然后在需要的时候再运行，运行的时候再定义session"
            "测试时相当于从头开始走测试图，测试的数据也是从头开始"

            #opt, grads_and_vars = self.compute_gradients(self.cost, gradient_clipping=1)
            #self.optimizer = opt.apply_gradients(grads_and_vars)

            #self.r_pred = self.test_pred(test_prior_weight, test_user_embedding_new, self.pred_seq)
            #self.skip_states = train_updated_states
            # self.proba_prediction = tf.matmul(x, output_fc_W) + softmax_b
        '''