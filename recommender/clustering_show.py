import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from recommender.item_clustering_layer import DeepTemporalClustering
import tensorflow as tf
import numpy as np
import tensorflow as tf
import random
from recommender.BasicRcommender_soft import BasicRecommender_soft
import time
#from component.RNN import RNN_Compoment
#from component.MLP import MLP
from rnn_cells.skip_rnn_cells import MultiSkipLSTMCell, MultiSkipGRUCell, SkipGRUCell, SkipLSTMCell
from rnn_cells.basic_rnn_cells import BasicGRUCell, BasicLSTMCell
import tensorflow.contrib.layers as layers
from recommender.item_clustering_layer import DeepTemporalClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm



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
        self.drop_memory = config['drop_memory']
        self.drop_user = config['dropout_user']
        self.dropout_context1 = config['dropout_context1']
        self.dropout_context2 = config['dropout_context2']
        self.cell_type = 'gru'
        self.single_predict = config['single_predict']

        self.rnn_unit_num_list = config['rnn_unit_num']
        self.rnn_unit_num = self.rnn_unit_num_list[-1]
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
        self.u_id_test = tf.placeholder(tf.int32, [self.testBatchSize, 1])

        self.target_seq_pos = tf.placeholder(tf.int32, [None, self.seq_length])
        self.target_seq_neg = tf.placeholder(tf.int32, [None, self.neg_num * self.seq_length])
        self.pred_seq = tf.placeholder(tf.int32, [None, self.eval_item_num])
        self.dropout_keep_placeholder = tf.placeholder_with_default(1.0, shape=())
        self.input_seq_rating_holder = tf.placeholder(tf.float32, [self.trainBatchSize, self.seq_length])
        self.target_seq_rating_holder = tf.placeholder(tf.float32, [self.trainBatchSize, self.seq_length])


        # user/item embedding
        self.itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 0.1))
        self.userEmbedding = tf.Variable(tf.random_normal([self.numUser, self.numFactor], 0, 1 / tf.sqrt(float(self.numFactor))))
        self.loss_type = config['loss_type']
        self.target_weight = config['target_weight']
        self.center_embedding_weight = tf.Variable(tf.constant(0.5, shape=[1, self.numFactor]))
        self.center_feature_weight = 0.3

        self.cost_per_sample = 1
        self.skip_states = None
        self.user_embedding_weight = tf.Variable(tf.constant(0.5, shape=[1, self.numFactor]))
        self.item_center_embedding_weight = tf.Variable(tf.constant(0.5, shape=[1, self.numFactor]))

        self.n_clusters = 15
        self.clustering_layer = None
        self.p_value = None

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
        self.cell, self.initial_state = self.create_rnn_cell_init_state()

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

    'create the target rnn cell and init state'

    def create_rnn_cell_init_state(self, learn_initial_state=True):
        """
            Returns a tuple of (cell, initial_state) to use with dynamic_rnn.
            If self.rnn_hidden_size is an integer, a single RNN cell will be created. If it is a list, a stack of len(self.rnn_hidden_size)
            cells will be created.
            """
        if not self.cell_type in ['lstm', 'gru', 'skip_lstm', 'skip_gru']:
            raise ValueError('The specified model is not supported. Please use {lstm, gru, skip_lstm, skip_gru}.')
        if isinstance(self.rnn_unit_num_list, list) and len(self.rnn_unit_num_list) > 1:
            if self.cell_type == 'skip_lstm':
                cells = MultiSkipLSTMCell(self.rnn_unit_num_list)
            elif self.cell_type == 'skip_gru':
                cells = MultiSkipGRUCell(self.rnn_unit_num_list)
            elif self.cell_type == 'lstm':
                cell_list = [BasicLSTMCell(n) for n in self.rnn_unit_num_list]
                cells = tf.contrib.rnn.MultiRNNCell(cell_list)
            elif self.cell_type == 'gru':
                cell_list = [BasicGRUCell(n) for n in self.rnn_unit_num_list]
                cells = tf.contrib.rnn.MultiRNNCell(cell_list)
            if learn_initial_state:
                if self.cell_type == 'skip_lstm' or self.cell_type == 'skip_gru':
                    initial_state = cells.trainable_initial_state(self.trainBatchSize)
                else:
                    initial_state = []
                    for idx, cell in enumerate(cell_list):
                        with tf.variable_scope('layer_%d' % (idx + 1)):
                            initial_state.append(cell.trainable_initial_state(self.trainBatchSize))
                    initial_state = tuple(initial_state)
            else:
                initial_state = None
            return cells, initial_state
        else:
            if isinstance(self.rnn_unit_num_list, list):
                self.rnn_unit_num = self.rnn_unit_num_list[0]
            if self.cell_type == 'skip_lstm':
                cell = SkipLSTMCell(self.rnn_unit_num)
            elif self.cell_type == 'skip_gru':
                cell = SkipGRUCell(self.rnn_unit_num)
            elif self.cell_type == 'lstm':
                cell = BasicLSTMCell(self.rnn_unit_num)
            elif self.cell_type == 'gru':
                cell = BasicGRUCell(self.rnn_unit_num)
            if learn_initial_state:
                initial_state = cell.trainable_initial_state(self.trainBatchSize)
            else:
                initial_state = None
            return cell, initial_state

    def using_skip_rnn(self):
        """
        Helper function determining whether a Skip RNN models is being used
        """
        return self.cell_type.lower() == 'skip_lstm' or self.cell_type.lower() == 'skip_gru'

    def split_rnn_outputs(self, rnn_outputs):
        """
        Split the output of dynamic_rnn into the actual RNN outputs and the state update gate
        """
        if self.using_skip_rnn():
            return rnn_outputs.h, rnn_outputs.state_gate
        else:
            return rnn_outputs, tf.no_op()

    def compute_budget_loss(self, updated_states, loss):
        """
        Compute penalization term on the number of updated states (i.e. used samples)
        """
        # return tf.reduce_mean(tf.reduce_sum(self.cost_per_sample * updated_states, 1), 0)
        if self.using_skip_rnn():
            return tf.reduce_mean(tf.reduce_sum(self.cost_per_sample * updated_states, 1), 0)
        else:
            return tf.zeros(loss.get_shape())

    def compute_gradients(self, loss, gradient_clipping=-1):
        """
        Create optimizer, compute gradients and (optionally) apply gradient clipping
        """
        opt = tf.train.AdamOptimizer(self.learnRate)
        if gradient_clipping > 0:
            vars_to_optimize = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, vars_to_optimize), clip_norm=gradient_clipping)
            grads_and_vars = list(zip(grads, vars_to_optimize))
        else:
            grads_and_vars = opt.compute_gradients(loss)
        return opt, grads_and_vars

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
        finished_predict = tf.nn.sigmoid(tf.matmul(reshape_finished_embedding, prediction_layer) + prediction_layer_bias)
        islike_predict = tf.nn.sigmoid(tf.matmul(reshape_islike_embedding, prediction_layer) + prediction_layer_bias)
        iscom_predict = tf.nn.sigmoid(tf.matmul(reshape_iscom_embedding, prediction_layer) + prediction_layer_bias)
        isfollow_predict = tf.nn.sigmoid(tf.matmul(reshape_isfollow_embedding, prediction_layer) + prediction_layer_bias)

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

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            #categories = list(categories)
            '''
                       colors = ['black', 'green', 'red', 'sienna','aliceblue','yellow','gold','hotpink','pink',
                                 'yellow','yellowgreen','lightyellow','deeppink','indigo','lightseagreen',
                                 'maroon','mediumpurple','orange']
                       color = {0: 'black', 1: 'green', 2: 'red', 3: 'sienna', 4: 'aliceblue', 5: 'yellow',
                                    6: 'gold', 7: 'hotpink', 8: 'pink', 9: 'yellow',10:'yellowgreen',11:'lightyellow',
                                    12:'deeppink',13:'indigo',14:'lightseagreen',15:'maroon',16:'mediumpurple',17:'orange'
                                    ,18:'snow'}
                       color = {r'Action\n': 'black', r'Adventure\n': 'green', r'Animation\n': 'red', r'Children\n': 'sienna', r'Comedy\n': 'aliceblue', r'Comedy\n': 'yellow',
                                    r'Documentary\n': 'gold', r'Drama\n': 'hotpink', r'Fantasy\n': 'pink',r'Film-Noir\n' : 'yellow', r'Horror\n': 'yellowgreen', r'Musical\n': 'lightyellow',
                                    r'Mystery\n': 'deeppink', r'Romance\n': 'indigo', r'Sci-Fi\n': 'lightseagreen', r'Thriller\n': 'maroon', r'War\n': 'mediumpurple',
                                    r'Western\n': 'orange'}
                       color = {r'Action': 'black', r'Adventure': 'green', r'Animation': 'red', r'Children': 'sienna', r'Comedy': 'aliceblue', r'Comedy': 'yellow',
                                    r'Documentary': 'gold', r'Drama': 'hotpink', r'Fantasy': 'pink',r'Film-Noir' : 'yellow', r'Horror': 'yellowgreen', r'Musical': 'lightyellow',
                                    r'Mystery': 'deeppink', r'Romance': 'indigo', r'Sci-Fi': 'lightseagreen', r'Thriller': 'maroon', r'War': 'mediumpurple',
                                    r'Western': 'orange'}
                       

            def change_color_label(label):
                # label = transfer_labels(label)
                color = {'Action': 'black', 'Adventure': 'green', 'Animation': 'red', 'Children': 'sienna', 'Comedy': 'aliceblue', 'Comedy': 'yellow',
                                    'Documentary': 'gold', 'Drama': 'hotpink', 'Fantasy': 'pink','Film-Noir' : 'yellow', 'Horror': 'yellowgreen', 'Musical': 'lightyellow',
                                    'Mystery': 'deeppink', 'Romance': 'indigo', 'Sci-Fi': 'lightseagreen', 'Thriller': 'maroon', 'War': 'mediumpurple',
                                    'Western': 'orange'}

                #assert len(np.unique(label)) <= len(color)
                return (color[i] for i in label)
            #print(labels[i].value)
            print(categories[0])
            print(color['Action'])
            
            
            color = {'Action': 'black', 'Adventure': 'green', 'Animation': 'red', 'Children': 'sienna',
                         'Comedy': 'aliceblue', 'Crime': 'yellow',
                         'Documentary': 'gold', 'Drama': 'hotpink', 'Fantasy': 'pink', 'Film-Noir': 'yellow',
                         'Horror': 'yellowgreen', 'Musical': 'lightyellow',
                         'Mystery': 'deeppink', 'Romance': 'indigo', 'Sci-Fi': 'lightseagreen', 'Thriller': 'maroon',
                         'War': 'mediumpurple',
                         'Western': 'orange'}

            '''
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            # 转化为numpy数组
            itemEmbedding_numpy = self.itemEmbedding.eval(session=sess)
            #all_item_embedding = sess.run(self.itemEmbedding)
            #np.savetxt(all_item_embedding)
            df = pd.read_csv("D:\yanjiusheng\DCSR\dataset\movies-labels.csv")
            #print(itemEmbedding_numpy)
            labels = np.array(df.loc[:, 'labels'])
            categories = np.unique(labels)
            data_tsne = TSNE(n_components=2, perplexity=30).fit_transform(itemEmbedding_numpy)

            '''
            itemEmbedding_x = []
            itemEmbedding_y = []
            for index, label in enumerate(labels):
                if (label == 'Western\n'):
                    itemEmbedding_x.append(data_tsne[index, 0])
                    itemEmbedding_y.append(data_tsne[index, 1])
            print(itemEmbedding_x)
            print(itemEmbedding_y)

                #arr = [1, 2, 5, 8, 9, 6, 7, 3, 4]
            # 求均值
            arr_mean_x = np.mean(itemEmbedding_x)
            arr_mean_y = np.mean(itemEmbedding_y)
            # 求方差
            arr_var_x = np.var(itemEmbedding_x)
            arr_var_y = np.var(itemEmbedding_y)
            # 求标准差
            arr_std_x = np.std(itemEmbedding_x, ddof=1)
            arr_std_y = np.std(itemEmbedding_y, ddof=1)
            # 求最值

            arr_New_x = sorted(itemEmbedding_x)
            arr_Min_x = arr_New_x[0]
            arr_Max_x = arr_New_x[len(arr_New_x) - 1]
            arr_New_y = sorted(itemEmbedding_y)
            arr_Min_y = arr_New_y[0]
            arr_Max_y = arr_New_y[len(arr_New_y) - 1]



            print("平均值为：x:%f" %arr_mean_x)
            print("平均值为：y:%f" %arr_mean_y)
            print("方差为：x:%f" %arr_var_x)
            print("方差为：y:%f" %arr_var_y)
            print("标准差为:x:%f" %arr_std_x)
            print("标准差为:y:%f" %arr_std_y)
            print("最小值是：x:%f", arr_Min_x)
            print("最小值是：y:%f", arr_Min_y)
            print("最大值是：x:%f", arr_Max_x)
            print("最大值是：y:%f", arr_Max_y)
            '''

            def change_color_label(label):
                # label = transfer_labels(label)
                color = {'Action': 'blue', 'Adventure': 'green', 'Animation': 'green', 'Children': 'sienna',
                         'Comedy': 'blue', 'Crime': 'yellow',
                         'Documentary': 'gold', 'Drama': 'hotpink', 'Fantasy': 'yellow', 'Film-Noir': 'blue',
                         'Horror': 'yellowgreen', 'Musical': 'green',
                         'Mystery': 'deeppink', 'Romance': 'indigo', 'Sci-Fi': 'lightseagreen', 'Thriller': 'maroon',
                         'War': 'mediumpurple',
                         'Western': 'orange'}
                return [color[i[:-1]] for i in label if ((i=='War\n')|(i=='Western\n')|(i=='Fantasy\n')
                                                         )|(i=='Film-Noir\n')|(i=='Musical\n')]
 
            index_choice=[]

            for index, label in enumerate(labels):
                if ((label=='War\n')|(label=='Western\n')|(label=='Fantasy\n')|(label=='Film-Noir\n')|(label=='Musical\n')
                                                         ):
                    index_choice.append(index)

            itemEmbedding_numpy_new = []
            for i in index_choice:
                itemEmbedding_numpy_new.append(itemEmbedding_numpy[i])
                    


            for i in range(200):
                data_tsne = TSNE(n_components=2, perplexity=30).fit_transform(itemEmbedding_numpy_new)
                plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=change_color_label(labels), marker='.', s=20)
                plt.savefig(r'C:\Users\84306\Desktop\clustering\clustering' +'/'+ str(i) +'.png')
                plt.close()




                #plt.savefig(r'C:\Users\84306\Desktop\clustering\Documentary'+'/'+str(i)+'.png')
                #plt.close()






            #data_tsne = TSNE(n_components=2, perplexity=30).fit_transform(itemEmbedding_numpy_new)
            '''

            data_tsne_new_x=[]
            data_tsne_new_y=[]

            for i in index_choice:
                data_tsne_new_x.append(data_tsne[i,0])
                data_tsne_new_y.append(data_tsne[i,1])
     
            color = {'Action':'black', 'Adventure': 'green', 'Animation': 'red', 'Children': 'sienna', 'Comedy': 'aliceblue', 'Crime': 'yellow',
                                    'Documentary': 'gold', 'Drama': 'hotpink', 'Fantasy': 'pink','Film-Noir' : 'yellow', 'Horror': 'yellowgreen', 'Musical': 'lightyellow',
                                    'Mystery': 'deeppink', 'Romance': 'indigo', 'Sci-Fi': 'lightseagreen', 'Thriller': 'maroon', 'War': 'mediumpurple',
                                    'Western': 'orange'}

            for i in range(18):
                data_tsne = TSNE(n_components=2, perplexity=30).fit_transform(itemEmbedding_numpy)
                c = categories[i][:-1]
                print(c)
                plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=color[c], marker='.')
            plt.show()
            '''


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
            # test input
            test_item_embed_input = tf.nn.embedding_lookup(self.itemEmbedding, self.test_input_seq)
            'test seq uid '
            userEmbedding_test = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.u_id_test),
                                            [-1, self.numFactor])
            clustering_params = {
                'n_clusters': self.n_clusters,
                'alpha': 1,
                'embedding': tf.reshape(item_embed_input, shape=[-1, self.numFactor])
            }
            self.clustering_layer = DeepTemporalClustering(params=clustering_params)

            center_embedding = tf.nn.embedding_lookup(self.clustering_layer.mu, self.clustering_layer.pred)

            #center_embedding = tf.matmul(self.clustering_layer.p, self.clustering_layer.mu)

            center_embedding = tf.reshape(center_embedding, shape=[-1, self.seq_length, self.numFactor])


            q = self.clustering_layer.soft_assignment(
                embeddings=tf.reshape(item_embed_input, shape=[-1, self.numFactor]),
                cluster_centers=self.clustering_layer.mu)
            p = self.clustering_layer.tensor_target_distribution(q)

            '''
            with open('D:\yanjiusheng\DCSR\dataset\movies.csv', 'rb') as csvfile:
                reader = csv.reader(csvfile)
                labels = [row[2] for row in reader]
            '''



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

        # input_seq_batch, pos_seq_batch, neg_seq_batch, input_rating_batch, target_rating_batch = self.getTrainData(batchId)
        input_seq_batch, pos_seq_batch, neg_seq_batch, input_seq_user_batch = self.getTrainData(batchId)
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
                print(self.sess.run(self.clustering_layer.mu))

        self.optimizer.run(feed_dict={
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep,
            self.input_user_id: input_seq_user_batch,
            self.clustering_layer.p: self.p_value
        })

        loss = self.cost.eval(feed_dict={
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep,
            self.input_user_id: input_seq_user_batch,
            self.clustering_layer.p: self.p_value
        })


        # skip_states = self.skip_states.eval(feed_dict={
        #     self.input_seq: input_seq_batch,
        #     self.target_seq_pos: pos_seq_batch,
        #     self.target_seq_neg: neg_seq_batch,
        #     self.dropout_keep_placeholder: self.dropout_keep,
        #     self.input_user_id: input_seq_user_batch
        # })
        skip_states = None
        totalLoss += loss
        #loss_kl = tf.Session().run(self.clustering_layer.loss_kl)

        end = time.time()
        if epochId % 5 == 0 and batchId == 0:
            self.logger.info("----------------------------------------------------------------------")
            self.logger.info(
                "batchId: %d epoch %d/%d   batch_loss: %.4f   time of a batch: %.4f" % (
                    batchId, epochId, self.maxIter, totalLoss, (end - start)))
            #self.logger.info("KL.loss: %.4f" % (loss_kl))
            self.evaluateRanking(epochId, batchId)
        #return totalLoss, skip_states, loss_kl
        return totalLoss, skip_states

    def getTrainData(self, batchId):
        # compute start and end
        start = time.time()

        input_seq_user_batch = []
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
            self.test_input_seq: input_seq,
            self.pred_seq: target_seq,
            self.u_id_test: input_seq_uid
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

