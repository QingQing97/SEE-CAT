# import warnings
# warnings.filterwarnings('always')

# For debug
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

import numpy as np, random, math
from tqdm import tqdm
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from baselinedataloader import DialogLoader
from baselineModel import baselineModel, MaskedNLLLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

import time


## 在训练模型中很常见的操作(Get)
def seed_everything(seed):
    random.seed(seed)                          ##  random.seed()：使用 random() 生成的随机数将会是同一个
    np.random.seed(seed)                       ##  np.random.seed()：每次生成的随机数都相同
    torch.manual_seed(seed)                    ##  为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)               ##  为当前GPU设置随机种子；
    torch.cuda.manual_seed_all(seed)           ##  为所有的GPU设置种子;
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon):
    "Prepare optimizer"
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params":  ([p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]),
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return optimizer

def configure_dataloaders(dataset, classify, batch_size):
    "Prepare dataloaders"
    train_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_loss_mask.tsv'
    valid_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_loss_mask.tsv'
    test_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_loss_mask.tsv'


    train_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + f'train_prompted_utterances_{prec}_{succ}.txt',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_' + classify + '.tsv',
        train_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_speakers.tsv',
        batch_size,
        ## shuffle=True
        shuffle=False
    )
    
    valid_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + f'valid_prompted_utterances_{prec}_{succ}.txt',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_' + classify + '.tsv',
        valid_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_speakers.tsv',
        batch_size,
        shuffle=False
    )
    
    test_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + f'test_prompted_utterances_{prec}_{succ}.txt',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_' + classify + '.tsv',
        test_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_speakers.tsv',
        batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader, test_loader


def metric_helper(dataset, labels, preds, masks, losses, task_type):
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)

    # 各类别下结果 # 初始结果忘记加 sample_weight=masks -> NewReport
    report = classification_report(labels, preds, sample_weight=masks, digits=4)
    print(report)

    if dataset == 'dailydialog':
        if task_type == 'main':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
            avg_fscore2 = round(
                f1_score(labels, preds, sample_weight=masks, average='weighted', labels=[0, 2, 3, 4, 5, 6]) * 100,
                2)  # 去除中性类别
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro') * 100, 2)
            avg_fscore4 = round(
                f1_score(labels, preds, sample_weight=masks, average='micro', labels=[0, 2, 3, 4, 5, 6]) * 100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)
            avg_fscore6 = round(
                f1_score(labels, preds, sample_weight=masks, average='macro', labels=[0, 2, 3, 4, 5, 6]) * 100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3, avg_fscore4, avg_fscore5, avg_fscore6]
        elif task_type == 'sub':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro') * 100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)
            fscores = [avg_fscore1, avg_fscore3, avg_fscore5]
    
    else:
        avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
        avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro') * 100, 2)
        avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)
        fscores = [avg_fscore1, avg_fscore3, avg_fscore5]
        
    return avg_loss, avg_accuracy, fscores, labels, preds, masks

                                               ## 训练或验证的算法 
def train_or_eval_model(dataset, mode, model, main_loss_function, dataloader, epoch, acc_steps, optimizer=None, train=False, grad_acc=False):
    losses1, preds1, labels1, masks1= [], [], [], []

    assert not train or optimizer!=None        ## 确保train和optimizer都不为None
    
    if train:                                  ## train/eval模式选择
        model.train()
    else:
        model.eval()

    all_features = []
    for utterances, labels, loss_masks, _ in tqdm(dataloader, leave=False):    

        utterances = [[x] for x in utterances]
        labels = [[x] for x in labels]
        loss_masks = [[x] for x in loss_masks]

        
        # create umask and qmask            
        lengths = [len(item) for item in utterances]
        ## 句子数 X 最长句子的长度：utterance-mask
        # umask = torch.zeros(len(lengths), max(lengths)).long().cuda()
        # ## 用1标记句子的每个单词
        # for j in range(len(lengths)):
        #     umask[j][:lengths[j]] = 1
       
        label1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in labels], batch_first=True).cuda()

        loss_masks = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_masks], batch_first=True).long().cuda()
        # 等同于自己制作的umask

        # obtain log probabilities
        if train:
            features, log_prob1 = model(utterances, lengths)
        else:
            with torch.no_grad():
                features, log_prob1 = model(utterances, lengths)

        lp1_ = log_prob1.transpose(0, 1).contiguous().view(-1, log_prob1.size()[2])
        labels1_ = label1.view(-1)
        loss1 = main_loss_function(lp1_, labels1_, loss_masks)
        pred1_ = torch.argmax(lp1_, 1)
        preds1.append(pred1_.data.cpu().numpy())
        labels1.append(labels1_.data.cpu().numpy())
        masks1.append(loss_masks.view(-1).cpu().numpy())
        losses1.append(loss1.item() * masks1[-1].sum())

        loss = loss1 

        if train:                           ## 反向传播
            if grad_acc:
                accumulation_steps = int(acc_steps)
                loss = loss/accumulation_steps
                loss.backward()
                if ((i + 1) % accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                i = i + 1
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        for feature in features:
            # feature shape: batch_size x 1024
            all_features.append(feature)
        
        # print(len(all_features),len(all_features[0]))

    return all_features, list(zip(metric_helper(dataset, labels1, preds1, masks1, losses1, 'main')))
                 
               ## valid_fscores1, test_fscores1, valid_losses1
def result_helper(valid_fscores, test_fscores, valid_losses, best_label, best_pred, best_mask, task_type):
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    print('\n')
    print('Test performance.')
    if dataset == 'dailydialog':
        if task_type == 'main':
            score1 = test_fscores[0][np.argmin(valid_losses)]
            score2 = test_fscores[0][np.argmax(valid_fscores[0])]
            score3 = test_fscores[1][np.argmin(valid_losses)]
            score4 = test_fscores[1][np.argmax(valid_fscores[1])]
            score5 = test_fscores[2][np.argmin(valid_losses)]
            score6 = test_fscores[2][np.argmax(valid_fscores[2])]
            score7 = test_fscores[3][np.argmin(valid_losses)]
            score8 = test_fscores[3][np.argmax(valid_fscores[3])]
            score9 = test_fscores[4][np.argmin(valid_losses)]
            score10 = test_fscores[4][np.argmax(valid_fscores[4])]
            score11 = test_fscores[5][np.argmin(valid_losses)]
            score12 = test_fscores[5][np.argmax(valid_fscores[5])]

            scores = [score1, score2, score3, score4, score5, score6,
                      score7, score8, score9, score10, score11, score12]
            scores_val_loss = [score1, score3, score5, score7, score9, score11]
            scores_val_f1 = [score2, score4, score6, score8, score10, score12]
            loss_at_epoch = np.argmin(valid_losses)
            f1_at_epoch = [np.argmax(valid_fscores[0]), np.argmax(valid_fscores[1]), np.argmax(valid_fscores[2]), np.argmax(valid_fscores[3]), \
                           np.argmax(valid_fscores[4]), np.argmax(valid_fscores[5])]

            res1 = 'Scores: Weighted, Weighted w/o Neutral, Micro, Micro w/o Neutral, Macro, Macro w/o Neutral'
            res2 = 'F1@Best Valid Loss: {}'.format(scores_val_loss)
            res3 = 'F1@Best Valid F1: {}'.format(scores_val_f1)
            res4 = 'loss at epoch:' + str(loss_at_epoch)
            res5 = 'F1 at epoch: {}'.format(f1_at_epoch)

            print(res1)
            print(res2)
            print(res3)
            print(res4)
            print(res5)
    
    
    ## 最优结果输出
    else:
        ## numpy.argmin(a, axis=None, out=None)[source]：Returns the indices of the minimum values along an axis. 
        ## By default, the index is into the flattened array, otherwise along the specified axis.
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        score3 = test_fscores[1][np.argmin(valid_losses)]
        score4 = test_fscores[1][np.argmax(valid_fscores[1])]
        score5 = test_fscores[2][np.argmin(valid_losses)]
        score6 = test_fscores[2][np.argmax(valid_fscores[2])]

        scores = [score1, score2, score3, score4, score5, score6]
        scores_val_loss = [score1, score3, score5]
        scores_val_f1 = [score2, score4, score6]
        loss_at_epoch = np.argmin(valid_losses)                  ## 用数组保存所有结果，返回最小值编号
        f1_at_epoch = [np.argmax(valid_fscores[0]), np.argmax(valid_fscores[1]), np.argmax(valid_fscores[2])]

        res1 = 'Scores: Weighted, Micro, Macro'
        res2 = 'F1@Best Valid Loss: {}'.format(scores_val_loss)
        res3 = 'F1@Best Valid F1: {}'.format(scores_val_f1)
        res4 = 'loss at epoch:' + str(loss_at_epoch)
        res5 = 'F1 at epoch: {}'.format(f1_at_epoch)
            
        print(res1)
        print(res2)
        print(res3)
        print(res4)
        print(res5)
        

if __name__ == '__main__':
    # Start timing
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--lr_decay_type', default='none', help="steplr|exlr")
    parser.add_argument('--lr_decay_param', default=0.5, type=float, help="steplr: 0.5|0.1;exlr:0.98|0.99|0.90")
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--model', default='roberta', help='which model family bert|roberta|xlnet')
    parser.add_argument('--mode', default='1', help='which mode 0: bert or roberta base | 1: bert or roberta large; \
                                                     0, 1: bert base, large sentence transformer and 2, 3: roberta base, large sentence transformer')
    parser.add_argument('--dataset', default='dailydialog')
    parser.add_argument('--grad_acc', action='store_true', default=False, help='use grad accumulation')
    parser.add_argument('--acc_steps', default='1', help='1|2|4|8')
    parser.add_argument('--seed', type=int, default=777, metavar='seed', help='seed')
    parser.add_argument('--describe', default='train.py')

    parser.add_argument('--save_model', action='store_true', default=False, help='save model')

    parser.add_argument('--prec', default=5, type=int, help='preceding utterance number')
    parser.add_argument('--succ', default=5, type=int, help='succeeding utterance number')

    args = parser.parse_args()

    print(args)

    global dataset                        ## global: 若想在函数内部对函数外的变量进行操作，就需要在函数内部声明其为global
    dataset = args.dataset

    global prec
    prec = args.prec
    global succ
    succ = args.succ


    D_h = 200 # lstm layer
    batch_size = args.batch_size
    n_epochs = args.epochs
    transformer_model = args.model
    transformer_mode = args.mode
    grad_acc = args.grad_acc
    acc_steps = args.acc_steps


    global seed
    seed = args.seed
    seed_everything(seed)                 ## seed_everything: 自定义函数; seed default=777
    
    if dataset == 'dailydialog':
        print ('Classifying emotion in dailydialog.')
        n_classes  = 7
    elif dataset == 'meld':
        print ('Classifying emotion in meld.')
        n_classes  = 7
    elif dataset == 'iemocap':
        print ('Classifying emotion in iemocap.')
        n_classes  = 6
        

    ## key part
    ## transformer_model = args.model = Bert
    model = baselineModel(transformer_model, transformer_mode, n_classes, False, False)

    '''
    anger	0 
    no_emotion	1 
    disgust	2 
    fear	3 
    happiness	4 
    sadness	5
    surprise	6
    '''
    main_loss_function = MaskedNLLLoss()   

    optimizer = configure_optimizers(model, args.weight_decay, args.lr, args.adam_epsilon)     ## 优化器的设计
    if args.lr_decay_type == 'none':
        pass
    elif args.lr_decay_type == 'exlr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.lr_decay_param)
    elif args.lr_decay_type == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay_param)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, valid_loader, test_loader = configure_dataloaders(dataset, 'emotion', batch_size)
    

    valid_losses1, valid_fscores1= [], []
    test_fscores1 = []
    best_loss1 = None
    best_label1, best_pred1, best_mask1  = [], [], []
    train_losses1,  test_losses1 = [], []

    # if args.save_model:
    saved_model_number = int(time.time() * 256)                                    ## 随机生成保存模型的编号
    print('saved_model_number is: ' + str(saved_model_number))


    best_train_features,  best_valid_features, best_test_features = [], [], []

    for e in range(n_epochs):
        start_time = time.time()                                               ## 记录程序开始的时间
        print('\n')       
        print('---------train--------')    ## 160行自定义train_or_eval_mode函数 ## e: epoch数  # xxx_loader用于加载数据集 ## acc_steps default='1' 含义存疑
        train_features, train_result = train_or_eval_model(dataset, 0, model, main_loss_function, train_loader, e, acc_steps, optimizer, True, grad_acc)
        print('-----------valid-----------')                                   ## 默认train=False
        valid_features, valid_result = train_or_eval_model(dataset, 1, model, main_loss_function, valid_loader, e, acc_steps)
        print('-----------test-----------')
        test_features, test_result = train_or_eval_model(dataset, 2, model, main_loss_function, test_loader, e, acc_steps)

        if args.lr_decay_type != 'none':
            print("第%d个epoch的学习率：%f" % (e, optimizer.param_groups[0]['lr']))
            scheduler.step()

        # main task result
        valid_losses1.append(valid_result[0][0])
        valid_fscores1.append(valid_result[2][0])
        test_losses1.append(test_result[0][0])
        test_fscores1.append(test_result[2][0])
        train_losses1.append(train_result[0][0])

        if best_loss1 == None or best_loss1 > valid_result[0][0]:             ## 更新最优loss
            best_loss1 = valid_result[0][0]
            best_label1.append(test_result[3][0])
            best_pred1.append(test_result[4][0])
            best_mask1.append(test_result[5][0])
            
            best_train_features = train_features
            best_valid_features = valid_features
            best_test_features = test_features


        x1 = 'Epoch {}'.format(e) + '\n' + 'train_loss {} train_acc {} train_fscore {}'.format(train_result[0][0], train_result[1][0], train_result[2][0]) + '\n' + \
            'valid_loss {} valid_acc {} valid_fscore {}'.format(valid_result[0][0], valid_result[1][0], valid_result[2][0]) + '\n' + \
            'test__loss {} test__acc {} test__fscore {}'.format(test_result[0][0], test_result[1][0], test_result[2][0]) + '\n' + \
            'time {}'.format(round(time.time() - start_time, 2))
        print(x1)

        # save model
        if args.save_model:         
            fscores_ = np.array(valid_fscores1).transpose()
            state = {'model': model.state_dict(), 'epoch': e, 'seed': seed}
            
            if dataset == 'dailydialog':
                for i in [1, 3, 5]:
                    if np.argmax(fscores_[i]) == np.size(fscores_, 1) - 1:
                        # print('---------save best fscore model--------')
                        torch.save(state, 'saved_models/' + str(saved_model_number) + '_valid_fscores' + str(i) + '.pth')
            else:
                for i in [0, 1, 2]:
                    if np.argmax(fscores_[i]) == np.size(fscores_, 1) - 1:
                        # print('---------save best fscore model--------')
                        torch.save(state, 'saved_models/' + str(saved_model_number) + '_valid_fscores' + str(i) + '.pth')
    

    best_train_features = np.array([feature.detach().cpu().numpy() for feature in best_train_features])
    best_valid_features = np.array([feature.detach().cpu().numpy() for feature in best_valid_features])
    best_test_features = np.array([feature.detach().cpu().numpy() for feature in best_test_features]) 
    
    # 保存 best_features 到文件
    best_features_file_path = f'Features/{dataset}/{saved_model_number}_best_train_features.pkl'  # 指定要保存的文件路径
    with open(best_features_file_path, 'wb') as f:
        print(len(best_train_features))
        pickle.dump(best_train_features, f)
    
    best_features_file_path = f'Features/{dataset}/{saved_model_number}_best_valid_features.pkl'  # 指定要保存的文件路径
    with open(best_features_file_path, 'wb') as f:
        print(len(best_valid_features))
        pickle.dump(best_valid_features, f)
    
    best_features_file_path = f'Features/{dataset}/{saved_model_number}_best_test_features.pkl'  # 指定要保存的文件路径
    with open(best_features_file_path, 'wb') as f:
        print(len(best_test_features))
        pickle.dump(best_test_features, f)
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Program executed in {execution_time:.2f} seconds.")

    ## 主任务结果
    result_helper(valid_fscores1, test_fscores1, valid_losses1, best_label1, best_pred1, best_mask1, 'main')
    



