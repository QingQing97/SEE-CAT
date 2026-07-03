import pickle
import sys

model_id = '442790004772'
trainpklfile = f'Features/meld/{model_id}_best_train_features.pkl'
validpklfile = f'Features/meld/{model_id}_best_valid_features.pkl'
testpklfile = f'Features/meld/{model_id}_best_test_features.pkl'

trainfeatures = pickle.load(open(trainpklfile, 'rb'))
validfeatures = pickle.load(open(validpklfile, 'rb'))
testfeatures = pickle.load(open(testpklfile, 'rb'))

traincountfile = f'datasets/dialogue_level_minibatch/meld/train_utterance_counts.csv'
validcountfile = f'datasets/dialogue_level_minibatch/meld/valid_utterance_counts.csv'
testcountfile = f'datasets/dialogue_level_minibatch/meld/test_utterance_counts.csv'
con2fea = {}

# 处理训练集
idx = 0
con_num = 0 
with open(traincountfile, 'r') as cf:
    next(cf)  # 跳过表头
    for line in cf:
        if con_num == 60:
            con_num += 1

        line = line.strip().split(',')
        utt_num = int(line[1])  # 第二列是 Utterance Count
        con2fea[con_num] = trainfeatures[idx: idx + utt_num]
        idx += utt_num
        con_num += 1 

# 处理验证集
idx = 0
with open(validcountfile, 'r') as cf:
    next(cf)  # 跳过表头
    for line in cf:
        line = line.strip().split(',')
        utt_num = int(line[1])  # 第二列是 Utterance Count
        con2fea[con_num] = validfeatures[idx: idx + utt_num]
        idx += utt_num
        con_num += 1 

# 处理测试集
idx = 0
with open(testcountfile, 'r') as cf:
    next(cf)  # 跳过表头
    for line in cf:
        line = line.strip().split(',')
        utt_num = int(line[1])  # 第二列是 Utterance Count
        con2fea[con_num] = testfeatures[idx: idx + utt_num]
        idx += utt_num
        con_num += 1 

print(con_num)  # 应该输出 1432

# print(con2fea[1151]) 
# print(con2fea[1152])

# 保存 con2fea 为 TextFeatures.pkl
output_file = 'MELD/TextFeatures.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(con2fea, f)

print(f"Features saved to {output_file}.")
