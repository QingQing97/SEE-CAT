import pickle
import sys

model_id = '442782170502'
trainpklfile = f'Features/iemocap/{model_id}_best_train_features.pkl'
validpklfile = f'Features/iemocap/{model_id}_best_valid_features.pkl'
testpklfile = f'Features/iemocap/{model_id}_best_test_features.pkl'

trainfeatures = pickle.load(open(trainpklfile, 'rb'))
validfeatures = pickle.load(open(validpklfile, 'rb'))
testfeatures = pickle.load(open(testpklfile, 'rb'))

traincountfile = f'datasets/dialogue_level_minibatch/iemocap/train_utterance_counts.csv'
validcountfile = f'datasets/dialogue_level_minibatch/iemocap/valid_utterance_counts.csv'
testcountfile = f'datasets/dialogue_level_minibatch/iemocap/test_utterance_counts.csv'
con2fea = {}

# 处理训练集
idx = 0
with open(traincountfile, 'r') as cf:
    next(cf)  # 跳过表头
    for line in cf:
        line = line.strip().split(',')
        con_id = line[0]  # 第一列是 Conversation ID
        utt_num = int(line[1])  # 第二列是 Utterance Count
        con2fea[con_id] = trainfeatures[idx: idx + utt_num]
        idx += utt_num

# 处理验证集
idx = 0
with open(validcountfile, 'r') as cf:
    next(cf)  # 跳过表头
    for line in cf:
        line = line.strip().split(',')
        con_id = line[0]  # 第一列是 Conversation ID
        utt_num = int(line[1])  # 第二列是 Utterance Count
        con2fea[con_id] = validfeatures[idx: idx + utt_num]
        idx += utt_num

# 处理测试集
idx = 0
with open(testcountfile, 'r') as cf:
    next(cf)  # 跳过表头
    for line in cf:
        line = line.strip().split(',')
        con_id = line[0]  # 第一列是 Conversation ID
        utt_num = int(line[1])  # 第二列是 Utterance Count
        con2fea[con_id] = testfeatures[idx: idx + utt_num]
        idx += utt_num

# 保存 con2fea 为 TextFeatures.pkl
output_file = 'IEMOCAP/TextFeatures.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(con2fea, f)

print(f"Features saved to {output_file}.")
