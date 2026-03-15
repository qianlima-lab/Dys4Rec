from recommender.dynamicLSTM import DynamicLSTM4RecRecommender
from data_model.SequenceDataModel import SequenceDataModel
import gc
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"



if __name__ == '__main__':

    # instant_video_5_core, digital_music_5_core_100, baby_5_core_100  apps_for_android_5 video_games_5

    config = {
        'generate_seq': True,
        'splitterType': 'userTimeRatio',
        'fileName': 'ml-100k',
        'trainType': 'test',
        'threshold': 0,
        'learnRate': 0.001,
        'maxIter': 1000,
        'trainBatchSize': 512,
        'testBatchSize': 512,
        'numFactor': 128,
        'topN': 5,
        'factor_lambda': 0.01,
        'goal': 'ranking',
        'verbose': False,
        'seq_length': 10,
        'input_length':10,
        'dropout_keep': 0.45,
        'dropout_item': 0.5,
        'dropout_context1': 0.5,
        'dropout_context2': 0.5,
        'dropout_user': 0.5,
        'drop_memory': 0.45,
        'rnn_unit_num': 128,
        'rnn_layer_num': 1,
        'rnn_cell': 'DynamicLSTM',
        'eval_item_num': 500,
        'seq_direc': 'hor',
        'early_stop': True,
        'random_seed': 123,
        'useRating': True,
        'loss_type': 'soft',
        'target_weight': 0.8,
        'numK': 1,
        'negative_numbers': 25,
        'familiar_user_num': 5,
        'need_process_data': False,
        'csv': True,
        'test_sparse_user': True,
        'merge_sparse_user': False,
        'khsoft': False,
        'save_path': 'saved_model',
        'save_model': False,
        'load_model':True,
        'using_model': 'soft+rl',

        # DynamicLSTM
        'n_actions': 5,
        'n_features': 10
    }

    for fileName in ['ml-100k']:
        config['fileName'] = fileName
        seq_length = config['seq_length']

        dataModel = SequenceDataModel(config)
        dataModel.buildModel()

        for seq_length in [10]:
            config['seq_length'] = seq_length
            recommender = DynamicLSTM4RecRecommender(dataModel, config)
            recommender.run()

        # for rnn_unit_num in [32, 64, 128, 256]:
        #     for rnn_layer_num in [1]:
        #         for dp_kp in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        #             config['dropout_keep'] = dp_kp
        #             config['rnn_unit_num'] = rnn_unit_num
        #             config['rnn_layer_num'] = rnn_layer_num