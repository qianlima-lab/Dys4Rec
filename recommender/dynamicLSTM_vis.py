import numpy as np
import tensorflow as tf
import random
from recommender.BasicRcommender_soft_o import BasicRecommender_soft
from DynamicLSTM.NumberPrediction.supercell_new import LSTMCell
from DynamicLSTM.NumberPrediction.RL_brain import PolicyGradient
import time
import os
#from component.RNN import RNN_Compoment
#from component.MLP import MLP

#from rnn_cells.skip_rnn_cells import MultiSkipLSTMCell, MultiSkipGRUCell, SkipGRUCell, SkipLSTMCell, Modify_SkipGRUCell
from rnn_cells.skip_atten_rnn_cells import MultiSkipLSTMCell, MultiSkipGRUCell, SkipGRUCell, SkipLSTMCell
from rnn_cells.basic_rnn_cells import BasicGRUCell, BasicLSTMCell
import tensorflow.contrib.layers as layers
import pickle


class DynamicLSTM4RecRecommender(BasicRecommender_soft):

    def __init__(self, dataModel, config):
        "继承后的重写语句"
        super(DynamicLSTM4RecRecommender, self).__init__(dataModel, config)
        self.name = 'GRU4Rec'

        self.numFactor = config['numFactor']
        self.factor_lambda = config['factor_lambda']
        self.seq_length = config['seq_length']
        self.dropout_keep = config['dropout_keep']
        self.dropout_item = config['dropout_item']
        self.drop_memory = config['drop_memory']
        self.drop_user = config['dropout_user']

        self.dropout_item_placeholder = tf.placeholder_with_default(1.0, shape=())
        self.dropout_c1_placeholder = tf.placeholder_with_default(1.0, shape=())
        self.dropout_c2_placeholder = tf.placeholder_with_default(1.0, shape=())

        self.dropout_context1 = config['dropout_context1']
        self.dropout_context2 = config['dropout_context2']

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

        self.train_users = dataModel.train_users
        self.train_sequences_input = dataModel.train_sequences_input
        self.train_sequences_user_input = dataModel.train_sequences_user_input
        self.train_sequences_target = dataModel.train_sequences_target

        # "每个 item 的得分"
        # self.input_seq_rating = dataModel.input_seq_rating
        # self.target_seq_rating = dataModel.target_seq_rating
        # self.user_pred_sequences_actions = dataModel.user_pred_sequences_actions
        self.user_pred_sequences = dataModel.user_pred_sequences
        self.test_user_input_id = dataModel.test_user_input_id

        self.user_pred_user_sequences = dataModel.user_pred_user_sequences
        # self.train_sequences_actions_input = dataModel.train_sequences_actions_input
        # self.train_sequences_actions_target = dataModel.train_sequences_actions_target

        self.trainSize = len(self.train_sequences_input)
        self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1

        # placeholders
        self.input_seq = tf.placeholder(tf.int32, [None, self.seq_length])
        self.input_user_id = tf.placeholder(tf.int32, [None, 1])

        self.test_input_seq = tf.placeholder(tf.int32, [None, self.seq_length])
        self.u_id_test = tf.placeholder(tf.int32, [None, 1])

        self.target_seq_pos = tf.placeholder(tf.int32, [None, self.seq_length])
        self.target_seq_neg = tf.placeholder(tf.int32, [None, self.neg_num * self.seq_length])
        self.pred_seq = tf.placeholder(tf.int32, [None, self.eval_item_num])
        self.dropout_keep_placeholder = tf.placeholder_with_default(1.0, shape=())
        self.input_seq_rating_holder = tf.placeholder(tf.float32, [self.trainBatchSize, self.seq_length])
        self.target_seq_rating_holder = tf.placeholder(tf.float32, [self.trainBatchSize, self.seq_length])

        # RL agent
        self.n_actions = config['n_actions']
        self.n_features = config['n_features']

        # user/item embedding
        self.itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 0.1))
        self.userEmbedding = tf.Variable(tf.random_normal([self.numUser, self.numFactor], 0, 1 / tf.sqrt(float(self.numFactor))))
        self.loss_type = config['loss_type']
        self.target_weight = config['target_weight']

        self.cost_per_sample = 1
        self.user_embedding_weight = tf.Variable(tf.constant(0.5, shape=[1, self.numFactor]))

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
        # self.cell = tf.contrib.rnn.GRUCell(self.rnn_unit_num)
        self.RL = PolicyGradient(self.n_actions, self.n_features)
        self.cell = LSTMCell(self.rnn_unit_num, self.RL)

        labels_vector1 = tf.constant(1.0, shape=[self.trainBatchSize * self.seq_length, self.numK, 1])
        labels_vector2 = tf.constant(0.0, shape=[self.trainBatchSize * self.seq_length, self.numK, self.neg_num])
        self.labels2 = tf.concat([labels_vector1, labels_vector2], axis=2)

        self.output_fc_W = tf.get_variable(
            name="output_fc_W",
            dtype=tf.float32,
            shape=[self.numK * self.numFactor, self.rnn_unit_num],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        self.denselayer = tf.get_variable("denselayer", shape=[self.numK * self.numFactor, self.numFactor+self.rnn_unit_num],
                                          initializer=tf.random_uniform_initializer(
                                              minval=-1 / tf.sqrt(float(self.numFactor)),
                                              maxval=1 / tf.sqrt(float(self.numFactor))),
                                          dtype=tf.float32, trainable=True)
        self.denseBias = tf.Variable(tf.random_normal([self.numK * self.numFactor], 0, 0.1))

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
        # self.proba_prediction = None

        # [[i0, i1, i2, ..., in], [a0, a1, a2, ..., an]]
        self.vis_res = []
        self.actions = None

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

    def buildModel(self):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.input_seq)
            item_embed_input = tf.nn.dropout(tf.reshape(item_embed_input, [-1, self.seq_length, self.numFactor]),
                                             self.dropout_item_placeholder)

            userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.input_user_id),
                                       [-1, self.numFactor])

            user_embedding_drop = tf.nn.dropout(userEmbedding, self.drop_user)
            # padding to shape [batch, input_seq, numFactor] in order to fusion the input item seq embedding
            user_padding_embedding = tf.tile(user_embedding_drop, [1, self.seq_length])
            user_padding_embedding = tf.reshape(user_padding_embedding, shape=[-1, self.seq_length, self.numFactor])

            # test input

            'test seq uid '
            userEmbedding_test = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.u_id_test),
                                            [-1, self.numFactor])
            user_padding_embedding_test = tf.tile(userEmbedding_test, [1, self.seq_length])
            user_padding_embedding_test = tf.reshape(user_padding_embedding_test, shape=[-1, self.seq_length, self.numFactor])
            fusion_embedding = tf.add(item_embed_input,
                                      tf.multiply(tf.clip_by_value(self.user_embedding_weight, 0.1, 1.0),
                                                  user_padding_embedding))
            #fusion_embedding = tf.add(item_embed_input, user_padding_embedding)
            fusion_embedding = tf.reshape(fusion_embedding, [-1, self.numFactor])
            item_embed_input = tf.nn.dropout(tf.reshape(fusion_embedding, [-1, self.seq_length, self.numFactor]),
                                             self.drop_memory)
            item_embed_input = tf.unstack(item_embed_input, axis=1)

            rnn_outputs, curr_state = tf.contrib.rnn.static_rnn(
                cell=self.cell,
                inputs=item_embed_input,
                dtype=tf.float32,
            )
            rnn_outputs = tf.transpose(rnn_outputs, perm=[1, 0, 2])

            split_outputs = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.rnn_unit_num]), self.dropout_c1_placeholder)
            gru_vector = tf.nn.dropout(tf.concat([tf.reshape(split_outputs, [-1, self.rnn_unit_num]),
                                                  tf.reshape(user_padding_embedding, [-1, self.numFactor])], axis=1),
                                       self.drop_user)
            gru_vector = tf.reshape(gru_vector, [-1, self.rnn_unit_num + self.numFactor])
            user_embedding_new = tf.reshape(tf.tanh(tf.matmul(gru_vector, self.denselayer, transpose_b=True)
                                                    + self.denseBias), [-1, self.numK, self.numFactor])

            prior_weight = tf.reshape(tf.matmul(split_outputs, self.prior_weight, transpose_b=True),
                                      [-1, self.numK, 1])

            # context_vector = tf.reshape(tf.tanh(tf.matmul(split_outputs, self.output_fc_W, transpose_b=True)
            #                                     + self.output_fc_b), [-1, self.numK, self.numFactor])

            context_drop = tf.nn.dropout(user_embedding_new, self.dropout_c2_placeholder)

            # fusion the final hidden state and user embedding
            test_split_outputs = tf.reshape(split_outputs, [-1, self.seq_length, self.rnn_unit_num])

            test_gru_vector = tf.reshape(test_split_outputs[:, -1:, :], [-1, self.rnn_unit_num])

            userEmbedding_test = tf.reshape(userEmbedding_test, [-1, self.numFactor])
            # test_dot_embedding = tf.multiply(test_gru_vector, userEmbedding_test)
            test_gru_vector = tf.concat([test_gru_vector, userEmbedding_test], axis=1)

            test_gru_vector = tf.reshape(test_gru_vector, [-1, self.rnn_unit_num + self.numFactor])
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

            #budget_loss = self.compute_budget_loss(train_update_states, self.cost)
            if self.loss_type == 'bpr':
                cost = self.get_bpr_pred(element_pos, element_neg)
            else:
                cost = self.get_soft_pred(prior_weight, element_pos, element_neg)
            # RL agent
            actions = tf.transpose(tf.squeeze(self.RL.actions))
            all_act_prob = tf.transpose(self.RL.all_act_prob, perm=(1, 0, 2))
            neg_log_prob = tf.reduce_sum(-tf.log(all_act_prob + 0.000001) * tf.one_hot(actions, self.n_actions),
                                         axis=2)
            # rewards = tf.subtract(tf.sigmoid(tf.cast(-cost, tf.float32)), 0.5) * 2  # todo
            rewards = tf.tanh(tf.cast(-cost, tf.float32))
            rewards = tf.expand_dims(rewards, -1)
            loss_RL = tf.reduce_mean(neg_log_prob * rewards)

            self.cost = cost + loss_RL
            self.r_pred = self.test_pred(test_prior_weight, test_user_embedding_new, self.pred_seq)

    @staticmethod
    def get_bpr_pred(element_pos, element_neg):
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

        test_item_embedding = tf.nn.embedding_lookup(self.itemEmbedding, test_item_ids)
        test_item_embedding = tf.reshape(test_item_embedding,
                                         [-1, self.eval_item_num, self.numFactor])
        if self.numK > 1:
            pred_soft = tf.nn.softmax(tf.reshape(tf.matmul(test_context_vector, test_item_embedding, transpose_b=True),
                                                 [-1,  self.numK, self.eval_item_num]), axis=2)
            pred_dot = tf.reshape(tf.reduce_sum(tf.multiply(pred_soft, test_prior_weight), axis=1),
                                  [-1, self.eval_item_num])
        else:
            pred_dot = tf.reshape(tf.matmul(test_context_vector, test_item_embedding, transpose_b=True),
                                  [-1, self.eval_item_num])
        return pred_dot

    def trainEachBatch(self, epochId, batchId):
        totalLoss = 0
        start = time.time()

        input_seq_batch, pos_seq_batch, neg_seq_batch, input_seq_user_batch = self.getTrainData(batchId)

        self.optimizer.run(feed_dict={
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep,
            self.dropout_item_placeholder: self.dropout_item,
            self.dropout_c1_placeholder: self.dropout_context1,
            self.dropout_c2_placeholder: self.dropout_context2,
            self.input_user_id: input_seq_user_batch
        })

        loss = self.cost.eval(feed_dict={
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep,
            self.dropout_item_placeholder: self.dropout_item,
            self.dropout_c1_placeholder: self.dropout_context1,
            self.dropout_c2_placeholder: self.dropout_context2,
            self.input_user_id: input_seq_user_batch
        })

        totalLoss += loss

        end = time.time()
        if epochId % 5 == 0 and batchId == 0:
            self.logger.info("----------------------------------------------------------------------")
            self.logger.info(
                "batchId: %d epoch %d/%d   batch_loss: %.4f   time of a batch: %.4f" % (
                    batchId, epochId, self.maxIter, totalLoss, (end - start)))

            self.evaluateRanking(epochId, batchId)
        return totalLoss

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

        # input_rating_batch = self.input_seq_rating[start_idx:end_idx]
        # target_rating_batch = self.target_seq_rating[start_idx:end_idx]

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

        input_seq_user_batch = np.array(user_batch).reshape(-1, 1)
        input_seq_batch = np.array(input_seq_batch)
        pos_seq_batch = np.array(pos_seq_batch)
        neg_seq_batch = np.array(neg_seq_batch)

        end = time.time()
        # self.logger.info("time of collect a batch of data: " + str((end - start)) + " seconds")
        # self.logger.info("batch Id: " + str(batchId))

        return input_seq_batch, pos_seq_batch, neg_seq_batch, input_seq_user_batch
               #input_rating_batch, target_rating_batch

    def getPredList_ByUserIdxList(self, user_idices):
        end0 = time.time()
        # build test batch
        input_seq = []
        target_seq = []
        input_seq_uid = []

        for userIdx in user_idices:
            input_seq.append(self.user_pred_sequences[userIdx])
            target_seq.append(self.evalItemsForEachUser[userIdx])
            input_seq_uid.append(userIdx)

        input_seq = np.array(input_seq)
        target_seq = np.array(target_seq)
        input_seq_uid = np.array(input_seq_uid).reshape(-1, 1)

        end1 = time.time()

        predList = self.sess.run(self.r_pred, feed_dict={
            self.input_seq: input_seq,
            self.pred_seq: target_seq,
            self.u_id_test: input_seq_uid,
            self.input_user_id: input_seq_uid
        })

        end2 = time.time()

        output_lists = []
        for i in range(len(user_idices)):
            recommendList = {}
            start = i * self.eval_item_num
            end = start + self.eval_item_num
            for j in range(end-start):
                recommendList[target_seq[i][j]] = predList[i][j]
            sorted_RecItemList = sorted(recommendList, key=recommendList.__getitem__, reverse=True)[0:self.topN]
            output_lists.append(sorted_RecItemList)
        end3 = time.time()

        return output_lists, end1 - end0, end2 - end1, end3 - end2

    def vis(self):
        self.printInfo()
        self.buildModel_vis()
        self.visualize()

    def visualize(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        cur_save_path = './' + self.save_path + '/' + self.using_model + '/' + self.fileName
        ckpt = tf.train.get_checkpoint_state(cur_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('model has been loaded\n')
        else:
            print('no model has been loaded\n')
            pass

        for batchId in range(self.trainBatchNum):
            self.visualizeEachBatch(batchId)
        tf.reset_default_graph()

    def visualizeEachBatch(self, batchId):

        input_seq_batch, pos_seq_batch, neg_seq_batch, input_seq_user_batch = self.getTrainData(batchId)

        actions = self.actions.eval(feed_dict={
            self.input_seq: input_seq_batch,
            self.input_user_id: input_seq_user_batch
        })
        self.vis_res = [[list(i), [-1] + list(j)] for i, j in zip(input_seq_batch, actions)]
        self.vis_res = [[[int(self.itemIdxToItemId[idx-1]) if idx > 0 else -1 for idx in vr[0]], vr[1]] for vr in self.vis_res]
        save_path = 'kVis'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        using_model = self.using_model
        file = self.fileName
        action = self.n_actions
        seqLen = self.seq_length
        dump_file = './{}/{}_{}_action={}_seqLen={}.pkl'.format(save_path, using_model, file, action, seqLen)
        pickle.dump(self.vis_res, open(dump_file, 'wb'))

    def buildModel_vis(self):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

            item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.input_seq)
            item_embed_input = tf.nn.dropout(tf.reshape(item_embed_input, [-1, self.seq_length, self.numFactor]),
                                             self.dropout_item_placeholder)

            userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.input_user_id),
                                       [-1, self.numFactor])

            user_embedding_drop = tf.nn.dropout(userEmbedding, self.drop_user)
            user_padding_embedding = tf.tile(user_embedding_drop, [1, self.seq_length])
            user_padding_embedding = tf.reshape(user_padding_embedding, shape=[-1, self.seq_length, self.numFactor])

            fusion_embedding = tf.add(item_embed_input,
                                      tf.multiply(tf.clip_by_value(self.user_embedding_weight, 0.1, 1.0),
                                                  user_padding_embedding))
            fusion_embedding = tf.reshape(fusion_embedding, [-1, self.numFactor])
            item_embed_input = tf.nn.dropout(tf.reshape(fusion_embedding, [-1, self.seq_length, self.numFactor]),
                                             self.drop_memory)
            item_embed_input = tf.unstack(item_embed_input, axis=1)

            _, _ = tf.contrib.rnn.static_rnn(
                cell=self.cell,
                inputs=item_embed_input,
                dtype=tf.float32,
            )
            self.actions = tf.transpose(tf.squeeze(self.RL.actions))