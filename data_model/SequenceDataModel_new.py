from data_model.BasicDataModel_ml_new import BasicDataModel
# from matplotlib import pyplot as plt
# import matplotlib.cm as cm
import numpy as np
import pandas as pd


class BatchData:

    def __init__(self):
        self.users = []
        self.input_seqs = []
        self.target = []


class SequenceDataModel(BasicDataModel):

    def __init__(self, config):
        super(SequenceDataModel, self).__init__(config)

        self.train_users = []
        self.train_sequences_input = []
        self.train_sequences_user_input = []
        self.train_sequences_rnn = []
        self.train_sequences_target = []
        self.train_sequences_negative = []
        self.train_pos = []
        self.input_seq_rating = []
        self.target_seq_rating = []
        self.test_user_input_id = []

        self.userIdxToPreSeq = {}
        self.user_pred_sequences = {}
        self.user_pred_user_sequences = {}
        self.user_pred_sequences_rnn = []
        self.test_pos = []
        self.itemIdxToPath = {}
        self.itemIdxToCode = {}
        self.userIdxToAddTestCase = {}
        self.userIdxToAddItemNum = {}

        self.user_pred_seq_rating = {}

        self.NodeMask = []
        self.max_pos = 0
        self.generate_seq = config['generate_seq']
        self.khsoft = config['khsoft']
        self.single_predict = config['single_predict']

    # def append_pad_item(self):
    #     # add a padding item 0
    #     self.numItem += 1
    #     for userIdx, items in self.user_items_train.items():
    #         new_items = []
    #         for item in items:
    #             new_items.append(item+1)
    #         self.user_items_train[userIdx] = new_items
    #
    #     for userIdx, items in self.user_items_test.items():
    #         new_items = []
    #         for item in items:
    #             new_items.append(item + 1)
    #         self.user_items_test[userIdx] = new_items

    def createNewHuffmanAndNodeMaskTable(self):
        """（1）有pad item, 所以itemIdx要改变
           （2）要填补编码，所以有pad node，所以所有path中的node idx 要加一
           （3）对于itemIdx = 0怎么处理？"""
        self.numNode += 1
        self.NodeMask = np.array([[1.0] for i in range(self.numNode)], dtype=np.float32)
        self.NodeMask[0][0] = 0.0
        self.itemIdxToPath[0] = [0 for i in range(self.max_codelen)]
        self.itemIdxToCode[0] = [0 for i in range(self.max_codelen)]
        for itemidx, iteminfor in self.itemIdxToItemInfor.items():
            path = [0 for i in range(self.max_codelen)]
            code = [0 for i in range(self.max_codelen)]
            codelen = iteminfor['len']
            path[:codelen] = iteminfor['path']
            code[:codelen] = iteminfor['code']
            for i in range(codelen):
                path[i] += 1
            self.itemIdxToPath[itemidx + 1] = path
            self.itemIdxToCode[itemidx + 1] = code

    def generate_sequences_rnn_hor(self, seq_length):
        self.train_users = []
        self.train_sequences_input = []
        self.train_sequences_input_rnn = []
        self.train_sequences_user_input = []
        self.train_sequences_target = []
        self.user_pred_sequences = {}
        self.user_pred_user_sequences = {}
        self.user_pred_sequences_rnn = []
        self.test_user_input_id = []

        self.logger.info("\nseq length: %d" % (seq_length))

        # build train seqences
        for userIdx, items in self.user_items_train.items():
            for seq in self.slide_window_stride(items, seq_length + 1, seq_length):
                item_past_userlist = []
                input_seq = seq[0:-1]
                if self.single_predict:
                    target_seq = seq[1:]
                else:
                    target_seq = [seq[-1]]

                for train_item in input_seq:
                    if train_item == 0:
                        item_past_user = [0] * self.familiar_user_num
                    else:
                        time_position = self.itemIdxToPastUserTimePosition[train_item][userIdx]
                        if (time_position - self.familiar_user_num) < 0:
                            item_past_user = [0] * self.familiar_user_num
                            item_past_user[:time_position] = self.itemIdxToPastUserIdx[train_item][:time_position]
                        else:
                            item_past_user = self.itemIdxToPastUserIdx[train_item][(time_position -
                                                                                    self.familiar_user_num):time_position]
                    item_past_userlist.append(item_past_user)

                self.train_users.append(userIdx)
                self.train_sequences_user_input.append(item_past_userlist)
                self.train_sequences_input.append(input_seq)
                self.train_sequences_target.append(target_seq)

        # build pred sequences
        for userIdx in range(1, self.numUser):
            items = self.user_items_train[userIdx]

            if len(items) < seq_length:
                pred_seq = [0] * seq_length
                pred_seq[-len(items):] = items
            else:
                pred_seq = items[-seq_length:]

            item_past_userlist = []
            for test_item in pred_seq:
                if test_item == 0:
                    item_past_user = [0] * self.familiar_user_num

                else:
                    time_position = self.itemIdxToPastUserTimePosition[test_item][userIdx]
                    if (time_position - self.familiar_user_num) < 0:
                        item_past_user = [0] * self.familiar_user_num
                        item_past_user[:time_position] = self.itemIdxToPastUserIdx[test_item][:time_position]
                    else:
                        item_past_user = self.itemIdxToPastUserIdx[test_item][(time_position -
                                                                               self.familiar_user_num):time_position]

                item_past_userlist.append(item_past_user)

            self.user_pred_sequences[userIdx] = pred_seq
            self.user_pred_user_sequences[userIdx] = item_past_userlist
            self.test_user_input_id.append(userIdx)

    def generate_sequences_hor(self, input_length, target_length):
        self.train_users = []
        # 预测序列，batch_size * 5 已知前面5个 预测后面1个
        self.train_sequences_input = []
        # 每点击一个 item 会选择也点击了这个 item 的 前 5 个User：batch_size * 5 * 5
        self.train_sequences_user_input = []
        # 目标序列 batch_size * 1 需要预测的那个 item
        self.train_sequences_target = []
        self.train_sequences_negative = []
        self.user_pred_sequences = {}
        self.user_pred_user_sequences = {}
        # 训练样本的最后n（n为预测目标的个数，此处为1）个 item
        self.train_sequences_rnn = []
        self.user_pred_sequences_rnn = []
        # 样本的得分
        self.input_seq_rating = []
        self.target_seq_rating = []
        # 记录滑窗位置信息
        self.train_pos = []
        self.test_pos = []

        self.max_pos = 0

        self.logger.info("input length: %d, target length: %d" %
                         (input_length, target_length))

        "用一个slide window来在序列上滑行，slide window的size包括用来预测的input_length,和label的size，为target_size:" \
        "用5个预测1个"

        seq_length = input_length + target_length

        # build train sequences
        for userIdx, items_rating in self.user_items_train.items():
            # items = items_rating[:][0]
            pos = 0

            for seq in self.slide_window(items_rating, seq_length):
                item_past_userlist = []
                input_seq = np.array(seq[0:input_length])[:, 0].reshape(-1).tolist()
                target_seq = np.array(seq[1:])[:, 0].reshape(-1).tolist()
                rnn_seq = np.array(seq[-target_length - 1: -1])[:, 0].tolist()

                input_seq_rate = np.array(seq[0:input_length])[:, 1].reshape(-1).tolist()
                target_seq_rate = np.array(seq[1:])[:, 1].reshape(-1).tolist()

                for train_item in input_seq:
                    if train_item == 0:
                        item_past_user = [0] * self.familiar_user_num
                    else:
                        time_position = self.itemIdxToPastUserTimePosition[train_item][userIdx]
                        if (time_position - self.familiar_user_num) < 0:
                            item_past_user = [0] * self.familiar_user_num
                            item_past_user[:time_position] = self.itemIdxToPastUserIdx[train_item][:time_position]
                        else:
                            item_past_user = self.itemIdxToPastUserIdx[train_item][(time_position -
                                                                                    self.familiar_user_num):time_position]
                    item_past_userlist.append(item_past_user)
                    "shape: [input_size, familiar_user_num]"

                self.train_users.append(userIdx)
                "train_users 和 numUser不一样， numUser和testSize比较相近"
                self.train_sequences_input.append(input_seq)
                self.train_sequences_user_input.append(item_past_userlist)
                self.train_sequences_target.append(target_seq)
                self.train_sequences_rnn.append(rnn_seq)
                self.train_pos.append(pos)

                self.input_seq_rating.append(input_seq_rate)
                self.target_seq_rating.append(target_seq_rate)

                pos += 1
                if pos > self.max_pos:
                    self.max_pos = pos

        # build pred sequences
        for userIdx in range(1, self.numUser):
            items = np.array(self.user_items_train[userIdx])[:, 0].tolist()
            rating = np.array(self.user_items_train[userIdx])[:, 1].tolist()
            "当items的数量不够时，使用item 0来进行填充"
            if len(items) < input_length:
                pred_seq = [0] * input_length
                pred_seq[-len(items):] = items
            else:
                if (self.increaseTestcase):
                    AddItemNum = self.userIdxToAddItemNum[userIdx]
                    pred_seq = items[-(input_length + AddItemNum):]
                    add_pred = []
                    if (AddItemNum == 0):
                        add_pred.append(pred_seq)
                        self.userIdxToPreSeq[userIdx] = add_pred
                    else:
                        "test_label的顺序"
                        for i in range(AddItemNum):
                            add_pred.append(pred_seq[i:-(AddItemNum - i)])
                        add_pred.append(pred_seq[-input_length:])
                        self.userIdxToPreSeq[userIdx] = add_pred
                        self.user_items_test[userIdx].extend(pred_seq[-AddItemNum:].reverse())
                        self.user_items_test[userIdx].reverse()
                        "user_items_test shape:  dict{key: userIdx,  value : [[label1],[label2],[label3]]}  "
                        self.user_items_test[userIdx] = [[i] for i in self.user_items_test[userIdx]]
                else:
                    pred_seq = items[-input_length:]

            item_past_userlist = []
            for test_item in pred_seq:
                if test_item == 0:
                    item_past_user = [0] * self.familiar_user_num
                else:
                    time_position = self.itemIdxToPastUserTimePosition[test_item][userIdx]
                    if (time_position - self.familiar_user_num) < 0:
                        item_past_user = [0] * self.familiar_user_num
                        item_past_user[:time_position] = self.itemIdxToPastUserIdx[test_item][:time_position]
                    else:
                        item_past_user = self.itemIdxToPastUserIdx[test_item][(time_position -
                                                                               self.familiar_user_num):time_position]

                item_past_userlist.append(item_past_user)
                "shape: [input_size, familiar_user_num]"
            self.user_pred_sequences[userIdx] = pred_seq
            self.user_pred_user_sequences[userIdx] = item_past_userlist
            self.user_pred_sequences_rnn.append(items[-1:])
            self.test_pos.append(len(items) - seq_length)
            self.user_pred_seq_rating[userIdx] = rating[-input_length:]

    def generate_sequences_rnn_ver(self, seq_length):
        self.train_sequences_input = []
        self.train_sequences_target = []
        self.user_pred_sequences = []

        # get max session length
        max_session_len = 0
        for userIdx, items in self.user_items_train.items():
            if len(items) > max_session_len:
                max_session_len = len(items)

        self.logger.info("\nseq length: %d" %
                         (seq_length))

        action_1 = []
        action_2 = []
        for j in range(max_session_len):
            for i in range(self.numUser):
                items = self.user_items_train[i]
                if len(items) >= j + 2:
                    action_1.append(items[j])
                    action_2.append(items[j + 1])

        for seq in self.slide_window(action_1, seq_length):
            self.train_sequences_input.append(seq)
        for seq in self.slide_window(action_2, seq_length):
            self.train_sequences_target.append(seq)

        # build pred sequences
        for userIdx in range(self.numUser):
            items = self.user_items_train[userIdx]

            if len(items) < seq_length:
                pred_seq = [0] * seq_length
                pred_seq[-len(items):] = items
            else:
                pred_seq = items[-seq_length:]
            self.user_pred_sequences.append(pred_seq)

        print('generate sequences rnn ver finished')

    # def generate_sequences_hor(self, input_length, target_length):
    #
    #     self.train_users = []
    #     self.train_sequences_input = []
    #     self.train_sequences_user_input = []
    #     self.train_sequences_target = []
    #     self.train_sequences_negative = []
    #     self.user_pred_sequences = {}
    #     self.user_pred_user_sequences = {}
    #     self.train_sequences_rnn = []
    #     self.user_pred_sequences_rnn = []
    #
    #     self.train_pos = []
    #     self.test_pos = []
    #
    #     self.max_pos = 0
    #
    #     self.logger.info("input length: %d, target length: %d" %
    #                      (input_length, target_length))
    #
    #     "用一个slide window来在序列上滑行，slide window的size包括用来预测的input_length,和label的size，为target_size"
    #     seq_length = input_length + target_length
    #
    #     # build train seqences
    #     for userIdx, items in self.user_items_train.items():
    #         pos = 0
    #         if (self.increaseTestcase):
    #             train_lastIndex = (len(items) - 10) // 10
    #             if (train_lastIndex <= 0):
    #                 new_items = items
    #                 self.userIdxToAddItemNum[userIdx] = 0
    #             elif(train_lastIndex > 2):
    #                 new_items = items[:-2]
    #                 self.userIdxToAddItemNum[userIdx] = 2
    #             else:
    #                 new_items = items[:-train_lastIndex]
    #                 self.userIdxToAddItemNum[userIdx] = train_lastIndex
    #             "总的test label = additemNum + 1(原有的test)"
    #
    #             for seq in self.slide_window(new_items, seq_length):
    #                 input_seq = seq[0:input_length]
    #                 target_seq = seq[-target_length:]
    #                 rnn_seq = seq[-target_length - 1: -1]
    #                 self.train_users.append(userIdx)
    #                 "train_users 和 numUser不一样， numUser和testSize比较相近"
    #                 self.train_sequences_input.append(input_seq)
    #                 self.train_sequences_target.append(target_seq)
    #                 self.train_sequences_rnn.append(rnn_seq)
    #                 self.train_pos.append(pos)
    #                 pos += 1
    #                 if pos > self.max_pos:
    #                     self.max_pos = pos
    #         else:
    #             for seq in self.slide_window(items, seq_length):
    #                 item_past_userlist = []
    #                 input_seq = seq[0:input_length]
    #                 target_seq = seq[-target_length:]
    #                 rnn_seq = seq[-target_length - 1: -1]
    #
    #                 for train_item in input_seq:
    #                     if train_item == 0:
    #                         item_past_user = [0] * self.familiar_user_num
    #                     else:
    #                         time_position = self.itemIdxToPastUserTimePosition[train_item][userIdx]
    #                         if (time_position - self.familiar_user_num) < 0:
    #                             item_past_user = [0] * self.familiar_user_num
    #                             item_past_user[:time_position] = self.itemIdxToPastUserIdx[train_item][:time_position]
    #                         else:
    #                             item_past_user = self.itemIdxToPastUserIdx[train_item][(time_position -
    #                                                                                 self.familiar_user_num):time_position]
    #                     item_past_userlist.append(item_past_user)
    #                     "shape: [input_size, familiar_user_num]"
    #
    #                 self.train_users.append(userIdx)
    #                 "train_users 和 numUser不一样， numUser和testSize比较相近"
    #                 self.train_sequences_input.append(input_seq)
    #                 self.train_sequences_user_input.append(item_past_userlist)
    #                 self.train_sequences_target.append(target_seq)
    #                 self.train_sequences_rnn.append(rnn_seq)
    #                 self.train_pos.append(pos)
    #                 pos += 1
    #                 if pos > self.max_pos:
    #                     self.max_pos = pos
    #
    #
    #     # build pred sequences
    #     for userIdx in range(1, self.numUser):
    #         items = self.user_items_train[userIdx]
    #         "当items的数量不够时，使用item 0来进行填充"
    #         if len(items) < input_length:
    #             pred_seq = [0] * input_length
    #             pred_seq[-len(items):] = items
    #         else:
    #             if(self.increaseTestcase):
    #                 AddItemNum = self.userIdxToAddItemNum[userIdx]
    #                 pred_seq = items[-(input_length + AddItemNum):]
    #                 add_pred = []
    #                 if(AddItemNum==0):
    #                     add_pred.append(pred_seq)
    #                     self.userIdxToPreSeq[userIdx] = add_pred
    #                 else:
    #                     "test_label的顺序"
    #                     for i in range(AddItemNum):
    #                         add_pred.append(pred_seq[i:-(AddItemNum-i)])
    #                     add_pred.append(pred_seq[-input_length:])
    #                     self.userIdxToPreSeq[userIdx] = add_pred
    #                     self.user_items_test[userIdx].extend(pred_seq[-AddItemNum:].reverse())
    #                     self.user_items_test[userIdx].reverse()
    #                     "user_items_test shape:  dict{key: userIdx,  value : [[label1],[label2],[label3]]}  "
    #                     self.user_items_test[userIdx] = [[i] for i in self.user_items_test[userIdx]]
    #             else:
    #                 pred_seq = items[-input_length:]
    #
    #         item_past_userlist = []
    #         for test_item in pred_seq:
    #             if test_item == 0:
    #                 item_past_user = [0] * self.familiar_user_num
    #             else:
    #                 time_position = self.itemIdxToPastUserTimePosition[test_item][userIdx]
    #                 if (time_position - self.familiar_user_num) < 0:
    #                     item_past_user = [0] * self.familiar_user_num
    #                     item_past_user[:time_position] = self.itemIdxToPastUserIdx[test_item][:time_position]
    #                 else:
    #                     item_past_user = self.itemIdxToPastUserIdx[test_item][(time_position -
    #                                                                            self.familiar_user_num):time_position]
    #
    #             item_past_userlist.append(item_past_user)
    #             "shape: [input_size, familiar_user_num]"
    #         self.user_pred_sequences[userIdx] = pred_seq
    #         self.user_pred_user_sequences[userIdx] = item_past_userlist
    #         self.user_pred_sequences_rnn.append(items[-1:])
    #         self.test_pos.append(len(items) - seq_length)

    def generate_seq_for_memory_network(self, batchSize):
        # self.train_users = []
        # self.train_sequences_input = []
        # self.train_sequences_target = []

        # 1. get each user's seq length and start idx in old seqs
        user_seq_len = {}
        user_start_idx = {}
        max_len = 0
        min_len = 100
        input_data_size = len(self.train_users)
        cur_user_idx = self.train_users[0]
        user_start_idx[cur_user_idx] = 0
        for i in range(input_data_size):
            userIdx = self.train_users[i]
            if userIdx != cur_user_idx:
                user_start_idx[userIdx] = i
                user_seq_len[cur_user_idx] = i - user_start_idx[cur_user_idx]
                if user_seq_len[cur_user_idx] > max_len:
                    max_len = user_seq_len[cur_user_idx]
                if user_seq_len[cur_user_idx] < min_len:
                    min_len = user_seq_len[cur_user_idx]
                cur_user_idx = userIdx

            if i == input_data_size - 1:
                user_seq_len[cur_user_idx] = i - user_start_idx[cur_user_idx]
                if user_seq_len[cur_user_idx] > max_len:
                    max_len = user_seq_len[cur_user_idx]
                if user_seq_len[cur_user_idx] < min_len:
                    min_len = user_seq_len[cur_user_idx]

        # 2. build each user's index seq
        user_set = user_seq_len.keys()
        user_idx_seq = {}
        self.logger.info("max_len: " + str(max_len) + ", min_len: " + str(min_len))

        for userIdx in user_set:
            seq_len = user_seq_len[userIdx]
            seq = [i for i in range(seq_len)]

            copy_time = max_len // seq_len
            remainder = max_len % seq_len
            if remainder > 0:
                idx_seq = seq * copy_time + seq[-remainder:]
            else:
                idx_seq = seq * copy_time
            user_idx_seq[userIdx] = idx_seq

        # 3. build new seqs, collect each user's end indices
        batch_num = max_len
        new_train_users = []
        new_train_seqs_input = []
        new_train_seqs_target = []

        user_end_index = {}
        for userIdx in user_set:
            user_end_index[userIdx] = []

        index = 0
        for i in range(batch_num):
            for userIdx in user_set:
                shift = user_idx_seq[userIdx][i]
                start = user_start_idx[userIdx]

                if (i + 1) % user_seq_len[userIdx] == 0:
                    user_end_index[userIdx].append(index)

                input_seq = self.train_sequences_input[start + shift]
                target_seq = self.train_sequences_target[start + shift]

                new_train_users.append(userIdx)
                new_train_seqs_input.append(input_seq)
                new_train_seqs_target.append(target_seq)
                index += 1

        self.train_users = new_train_users
        self.train_sequences_input = new_train_seqs_input
        self.train_sequences_target = new_train_seqs_target

        whole_len = len(self.train_users)
        whole_index_list = [i for i in range(whole_len)]
        batch_idices = {}
        # # get batch_num
        if whole_len % batchSize == 0:
            batch_num = int(whole_len // batchSize)
        else:
            batch_num = int(whole_len // batchSize) + 1
        # # collect the indices of each batch
        for batchId in range(batch_num):
            start_idx = batchId * batchSize
            if start_idx + batch_num >= whole_len:
                end_idx = whole_len
            else:
                end_idx = start_idx + batchSize

            batch_idices[batchId] = [i + start_idx for i in range(start_idx, end_idx)]

        # collect each batch's end users
        batch_end_users = {}
        for batchId in range(batch_num):
            batch_end_users[batchId] = []
        for batchId in range(batch_num):
            curr_batch_idices = batch_idices[batchId]
            for userIdx in user_set:
                curr_user_end_idices = user_end_index[userIdx]
                for end_idx in curr_user_end_idices:
                    if end_idx in curr_batch_idices:
                        batch_end_users[batchId].append(int(userIdx))
                        break
        self.batch_end_users = batch_end_users

    def buildTestMatrix(self):
        for line in self.testSet:
            userIdx, itemIdx, rating = line
            # userIdx, itemIdx = line
            if self.generate_seq:
                itemIdx += 1
            self.testMatrix[userIdx, itemIdx] = rating
            # self.testMatrix[userIdx, itemIdx] = 1
        return self.testMatrix

    def slide_window(self, itemList, window_size):
        if len(itemList) < window_size:
            seq = [0] * window_size
            seq[-len(itemList):] = itemList
            yield seq
        else:
            num_seq = len(itemList) - window_size + 1
            for startIdx in range(num_seq):
                endIdx = startIdx + window_size
                seq = itemList[startIdx:endIdx]
                yield seq

    def slide_window_stride(self, itemList, window_size, stride):
        if len(itemList) < window_size:
            seq = [0] * window_size
            seq[-len(itemList):] = itemList
            yield seq
        else:
            num_seq = len(itemList) - window_size + 1
            startIdx = 0
            while startIdx < num_seq:
                endIdx = startIdx + window_size
                seq = itemList[startIdx:endIdx]
                startIdx += stride
                yield seq

    # def user_rating_stat(self):
    #
    #     # For plot configuration -----------------------------------------------------------------------------------
    #     fig, (ax1) = plt.subplots()
    #
    #     # Configure plot.
    #     plt.suptitle('user rating number statistic', fontsize=14, fontweight='bold')
    #
    #     # Configure 1st subplot.
    #     ax1.set_xlabel("rating number")
    #     ax1.set_ylabel("user")
    #     maxCount = 0
    #     self.user_item_num = [0] * self.numUser
    #     for userIdx, items in self.user_items_train.items():
    #         if len(items) > maxCount:
    #             maxCount = len(items)
    #         self.user_item_num[userIdx] = len(items)
    #     ax1.set_xlim([0, maxCount])
    #     ax1.set_ylim([0, self.numUser * (1.05)])
    #
    #     # For 1st subplot ------------------------------------------------------------------------------------------
    #
    #     # Plot Silhouette Coefficient for each sample
    #     y_lower = 10
    #
    #     ith_s = np.array(self.user_item_num)
    #     ith_s.sort()
    #     size_cluster_i = ith_s.shape[0]
    #     y_upper = y_lower + self.numUser
    #     ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_s, alpha=0.7)
    #     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(1))
    #     y_lower = y_upper + 10
    #
    #     # Plot the mean Silhouette Coefficient using red vertical dash line.
    #     ax1.axvline(x=sum(self.user_item_num) / len(self.user_item_num), color="red", linestyle="--")
    #
    #     plt.show()
    # def append_pad_item(self):
    #     # add a padding item 0
    #     self.numItem += 1
    #     for userIdx, items in self.user_items_train.items():
    #         new_items = []
    #         for item in items:
    #             new_items.append(item+1)
    #         self.user_items_train[userIdx] = new_items
    #
    #     for userIdx, items in self.user_items_test.items():
    #         new_items = []
    #         for item in items:
    #             new_items.append(item + 1)
    #         self.user_items_test[userIdx] = new_items
    def append_pad_item(self):
        # add a padding item 0
        self.numItem += 1
        for userIdx, items in self.user_items_train.items():
            new_items = []
            for item in items:
                new_items.append(item + 1)
            self.user_items_train[userIdx] = new_items

        for userIdx, items in self.user_items_test.items():
            new_items = []
            for item in items:
                new_items.append(item + 1)
            self.user_items_test[userIdx] = new_items

    def read_ml_1m_data(self):
        filepath = './dataset/ml-1m/ratings.dat'
        seq_data = pd.read_csv(filepath,
                               names=['fuin', 'feedid', 'rate', 'time_stamp'], sep='::', engine='python')

        '0 use to represent padding so the number of user and item +1'
        all_users = np.unique(seq_data.fuin)

        self.userSet = set(all_users)
        self.numUser = len(self.userSet) + 1
        self.itemSet = set(np.unique(seq_data.feedid))
        self.numItem = len(self.itemSet) + 1

        # format the index
        user_index = [i for i in range(1, self.numUser)]
        self.userIdToUserIdx = dict(zip(all_users, user_index))
        self.userIdxToUserId = dict(zip(user_index, all_users))
        self.userIdToUserIdx[0] = 'pad'
        self.userIdxToUserId['pad'] = 0

        item_index = [i for i in range(1, self.numItem)]
        self.itemIdToItemIdx = dict(zip(np.unique(seq_data.feedid), item_index))
        self.itemIdxToItemId = dict(zip(item_index, np.unique(seq_data.feedid)))
        self.userIdToUserIdx[0] = 'pad'
        self.userIdxToUserId['pad'] = 0

        self.trainMatrix = np.zeros(shape=[self.numUser, self.numItem], dtype=np.float32)

        for userId in all_users:
            userIdx = self.userIdToUserIdx[userId]
            "实际计算时都是采用Idx计算的，而不是采用Id"
            self.user_items_train.setdefault(userIdx, [])
            self.user_item_train_activity.setdefault(userIdx, [])
            self.user_items_test.setdefault(userIdx, [])
            'train data: series[0:-seq_length-1], test_data: series [seq_length-1: ]'
            target_user_watch_info = seq_data[seq_data.fuin == userId].sort_values(by='time_stamp')
            # transfer to feed index
            target_feed_index = [self.itemIdToItemIdx[feedid] for feedid in target_user_watch_info.feedid.values]

            seq_length = len(target_feed_index)
            # train_index = int(seq_length * 0.7)
            # test_index = int(seq_length * 0.8)
            train_index = seq_length - 2
            test_index = seq_length - 1

            target_rating = [rate for rate in target_user_watch_info.rate]
            'training_data '
            self.user_items_train[userIdx] = target_feed_index[:train_index]

            for i in range(len(self.user_items_train[userIdx])):
                train_item = self.user_items_train[userIdx][i]
                train_rating = target_rating[i]
                self.trainSet.append([userIdx, train_item, train_rating])

            # build val data
            self.user_items_val[userIdx] = target_feed_index[train_index:test_index]

            # build test data
            self.user_items_test[userIdx] = target_feed_index[test_index:]

            for i in range(len(self.user_items_test[userIdx])):
                test_item = self.user_items_test[userIdx][i]
                self.testSet.append([userIdx, test_item, target_rating[i + test_index]])

            if userIdx not in self.evalItemsForEachUser:
                self.evalItemsForEachUser[userIdx] = set()

            for itemIdx in self.user_items_test[userIdx]:
                self.itemsInTestSet.add(itemIdx)

            for itemIdx in set(target_feed_index):

                if itemIdx not in self.itemIdxToPastUserIdx.keys():
                    self.itemIdxToPastUserIdx[itemIdx] = [userIdx]
                    self.itemIdxToPastUserTimePosition[itemIdx] = {userIdx: 0}

                else:
                    self.itemIdxToPastUserTimePosition[itemIdx][userIdx] = len(
                        self.itemIdxToPastUserIdx[itemIdx])
                    self.itemIdxToPastUserIdx[itemIdx].append(userIdx)

        for line in self.trainSet:
            userIdx, itemIdx, rating = line
            self.trainMatrix[userIdx, itemIdx] = rating

    def buildModel(self):
        # self.pre_process()
        # self.split_UserTimeLOO(K=2)
        # self.logger.info("\n###### information of DataModel ######\n")
        if self.need_process_data:
            self.pre_process()
            self.split_UserTimeRatio()
            #self.sparse_split_UserTimeRatio()
        self.readData()
        self.append_pad_item()
        # self.read_ml_1m_data()

        if self.khsoft:
            self.createNewHuffmanAndNodeMaskTable()

        self.generateEvalItemsForEachUser()
        # self.generate_popular_EvalItemsForEachUser()
        "self.user_rating_stat()"
        self.printInfo()
