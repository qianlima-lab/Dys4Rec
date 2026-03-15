from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import time
item_full= {}
with open(r'/home/dev_data/chengyu/DIB/dataset/processed_datasets/movies_tv-seq/userTimeRatio/full.txt','r',encoding='ISO-8859-1') as file1:
    lines = file1.readlines()
    for i in lines:
        row = i.split(' ')
        if row[1] in item_full.keys():
            item_full[row[1]] = item_full[row[1]] + 1
        else:
            item_full[row[1]] = 1
file1.close()
user_select=[]
user_selec_x=[]

with open(r'/home/dev_data/chengyu/DIB/dataset/processed_datasets/movies_tv-seq/userTimeRatio/test.txt', 'r',
          encoding='ISO-8859-1') as file_test:
    lines = file_test.readlines()
    for line in lines:
        row = line.split(' ')
        if item_full[row[1]] < 6:
            user_select.append(row[0])
#print(user_select)
with open(r'/home/dev_data/chengyu/DIB/dataset/processed_datasets/movies_tv-seq/userTimeRatio/user_idx_id.txt', 'r',
          encoding='ISO-8859-1') as file_x:
    lines=file_x.readlines()[2:]
    #print(lines)
    for line in lines:
        line = line.strip('\n')
        row = line.split('   ')
        #print(row)
        if row[1] in user_select:
            user_selec_x.append(int(row[0]))
#print(user_selec_x)


class RankingEvaluator:

    def __init__(self, groundTruthLists, user_items_train, itemInTestSet, topK, testMatrix):

        '''
        :param groundTruthLists: a dict {userId:[itemId1, itemId2, ...], ...}
        :param user_items_train: a dict {userId:[itemId1, itemId2, ...], ...}
        :param itemInTestSet: Integer
        :param topK:    Integer
        :param testMatrix: a dict {(userId, itemId):rating ...}
        '''

        self.groundTruthLists = groundTruthLists
        self.user_items_train = user_items_train
        "predLists是个字典，key为userIdx，value为预测出的TopK个item list"
        self.predLists = None
        self.indexRange = len(self.groundTruthLists)
        self.itemInTestSet = itemInTestSet
        self.topK = topK
        self.userIdxToKind = {}
        self.H_userType = self.decideUserKind()
        self.testMatrix = testMatrix
        self.pool = Pool()

    def decideUserKind(self):
        H_userType = {'t0': 0, 't1': 0, 't2': 0, 't3': 0}
        for userIdx in self.groundTruthLists:
            num = len(self.user_items_train[userIdx])
            if num < 10:
                H_userType['t0'] += 1
                self.userIdxToKind[userIdx] = 't0'
            elif num < 25:
                H_userType['t1'] += 1
                self.userIdxToKind[userIdx] = 't1'
            elif num < 40:
                H_userType['t2'] += 1
                self.userIdxToKind[userIdx] = 't2'
            else:
                H_userType['t3'] += 1
                self.userIdxToKind[userIdx] = 't3'
        return H_userType

    def setPredLists(self, predLists):
        self.predLists = predLists

    '''Hit'''
    def calHit(self):
        start = time.time()
        results = []
        #Recall_userType = {'t0': 0, 't1': 0, 't2': 0, 't3': 0}
        for userIdx in user_selec_x:
            results.append(self.pool.apply_async(self.calculate_a_Recall, (userIdx, )))

        recallSum = 0
        for result in results:
            recallSum += result.get()

        end = time.time()
        return recallSum / len(results), (end - start)

    def calculate_a_Hit(self, userIdx):
        hitNum = 0
        userTrueList = self.groundTruthLists[userIdx]
        userPredList = self.predLists[userIdx]

        i = 0
        for itemIds in userPredList:
            if isinstance(itemIds, list):
                if userTrueList[i][0] in itemIds:
                    hitNum += 1
            else:
                if itemIds in userTrueList:
                    hitNum += 1
            i += 1

        return hitNum / len(userTrueList)

    '''Recall'''
    def calRecall(self):
        start = time.time()
        results = []
        Recall_userType = {'t0': 0, 't1': 0, 't2': 0, 't3': 0}
        #print(user_selec_x)

        for userIdx in user_selec_x:
            results.append(self.pool.apply_async(self.calculate_a_Recall, (userIdx, Recall_userType)))

        recallSum = 0
        for result in results:
            recallSum += result.get()

        r_0 = []
        r_1 = []
        r_2 = []
        r_3 = []

        total_user = len(results)
        total_hit  = recallSum

        if self.H_userType['t0']:
            t_0 = Recall_userType['t0'] / self.H_userType['t0']
        else:
            t_0 = -1

        r_0.extend([t_0, Recall_userType['t0'], self.H_userType['t0']])

        if self.H_userType['t1']:
            t_1 = Recall_userType['t1'] / self.H_userType['t1']
        else:
            t_1 = -1

        r_1.extend([t_1, Recall_userType['t1'], self.H_userType['t1']])

        if self.H_userType['t2']:
            t_2 = Recall_userType['t2'] / self.H_userType['t2']
        else:
            t_2 = -1

        r_2.extend([t_2, Recall_userType['t2'], self.H_userType['t2']])

        if self.H_userType['t3']:
            t_3 = Recall_userType['t3'] / self.H_userType['t3']
        else:
            t_3 = -1

        r_3.extend([t_3, Recall_userType['t3'], self.H_userType['t3']])

        end = time.time()
        return recallSum / len(results), (end - start), r_0, r_1, r_2, r_3, total_hit, total_user


    def calculate_a_Recall(self, userIdx, Recall_userType):
        hitNum = 0
        userTrueList = self.groundTruthLists[userIdx]
        userPredList = self.predLists[userIdx]

        i = 0
        for itemIds in userPredList:
            if isinstance(itemIds, list):
                if userTrueList[i][0] in itemIds:
                    hitNum += 1
            else:
                if itemIds in userTrueList:
                    hitNum += 1
            i += 1

        if hitNum:
            Recall_userType[self.userIdxToKind[userIdx]] += 1
        return hitNum / len(userTrueList)

    '''Precision'''
    def calPrecision(self):
        start = time.time()
        results = []

        for userIdx in user_selec_x:
            results.append(self.pool.apply_async(self.calculate_a_Precision, (userIdx,)))

        precisionSum = 0
        for result in results:
            precisionSum += result.get()

        end = time.time()

        return precisionSum / len(results), (end - start)


    def calculate_a_Precision(self, userIdx):
        hitNum = 0
        userTrueList = self.groundTruthLists[userIdx]
        userPredList = self.predLists[userIdx]

        i = 0
        for itemIds in userPredList:
            if isinstance(itemIds, list):
                for itemId in itemIds:
                    if itemId in userTrueList[i]:
                        hitNum += 1
            else:
                if itemIds in userTrueList:
                    hitNum += 1
            i += 1

        return hitNum / len(userPredList)

    '''AUC'''
    def calAUC(self):
        start = time.time()
        results = []

        for userIdx in user_selec_x:
            results.append(self.pool.apply_async(self.calculate_a_AUC, (userIdx,)))
        aucSum = 0
        for result in results:
            aucSum += result.get()

        end = time.time()

        return aucSum / len(results), (end - start)


    def calculate_a_AUC(self, userIdx):
        userTrueList = self.groundTruthLists[userIdx]
        userPredList = self.predLists[userIdx]

        numEval = len(self.itemInTestSet) - len(self.itemInTestSet.intersection(set(self.user_items_train[userIdx])))

        numRelevant = 0
        numMiss = 0

        for itemIdx in userTrueList:
            if itemIdx in userPredList:
                numRelevant += 1
            else:
                numMiss += 1

        if numEval == numRelevant:
            return 1.0

        numPairs = numRelevant * (numEval - numRelevant)

        if numPairs == 0:
            return 0.5

        numHits = 0
        numCorrectPairs = 0
        for itemIdx in userPredList:
            if itemIdx in userTrueList:
                numHits += 1
            else:
                numCorrectPairs += numHits

        numCorrectPairs += numHits * (numEval - len(userPredList) - numMiss)
        return numCorrectPairs / numPairs

    '''NDCG'''
    def calNDCG(self):
        start = time.time()
        results = []

        for userIdx in user_selec_x:
            results.append(self.pool.apply_async(self.calculate_a_NDCG, (userIdx,)))
        ndcgSum = 0
        for result in results:
            ndcgSum += result.get()

        end = time.time()

        return ndcgSum / len(results), (end - start)


    def calculate_a_NDCG(self, userIdx):
        ratingList = []

        userTrueList = self.groundTruthLists[userIdx]
        userPredList = self.predLists[userIdx]

        # calculate DCG and build rating list
        DCG = 0.0
        for index in range(len(userPredList)):
            itemIdx = userPredList[index]
            if itemIdx in userTrueList:
                ratingValue = self.testMatrix[userIdx, itemIdx]
                ratingList.append(ratingValue)
                DCG += ratingValue / np.log2(index + 2)
        if len(ratingList) == 0:
            return 0

        # calculate IDCG
        ratingList.sort(reverse=True)
        IDCG = 0.0
        for index in range(len(ratingList)):
            ratingValue = ratingList[index]
            IDCG += ratingValue / np.log2(index + 2)

        assert IDCG >= DCG
        return DCG / IDCG


if __name__ == '__main__':
    user_items_train = {0: [0, 1, 2, 3, 4, 5],
                        1: [0, 1, 3, 6],
                        2: [0, 2, 3, 6]}
    groundTruthList = {
                       0: [7, 8, 9],
                       1: [7, 8, 9],
                       2: [7, 8, 9]
    }
    topK = 3
    numItem = set()
    numItem.add(4)
    numItem.add(5)
    numItem.add(7)
    numItem.add(8)
    numItem.add(9)

    testMatrix = {(1, 7): 5,
                  (1, 8): 4,
                  (1, 9): 3,
                  (0, 7): 5,
                  (0, 8): 4,
                  (0, 9): 3,
                  (2, 7): 5,
                  (2, 8): 4,
                  (2, 9): 3,
                  }
    predLists = {0: [7, 8, 9],
                 1: [7, 4, 5],
                 2: [4, 7, 8],
                 }
    evaluator = RankingEvaluator(
        groundTruthLists=groundTruthList,
        user_items_train=user_items_train,
        itemInTestSet=numItem,
        topK=topK,
        testMatrix=testMatrix)
    evaluator.setPredLists(predLists)
    print("AUC:" + str(evaluator.calAUC()))
    print("Recall:" + str(evaluator.calRecall()))
    print("Precision:" + str(evaluator.calPrecision()))
    print("NDCG:" + str(evaluator.calNDCG()))









