import sys
import os
import pickle
import math
import time
import random
from random import choices
from collections import defaultdict


import torch
import torch.nn as nn
import numpy as np


TRAINING_LEN = 2000000
PREDICTION_LEN = 2000000
MB = 1000000
MAX_LIST = [0] * 22
MIN_LIST = [0] * 22

def gen_data(expert_0, expert_1):
    input = []
    labels = []
    prediction_input = []
    prediction_labels = []
    
    train_files = [path for path in os.listdir("/mydata/features/train-set")]
    # num_file = len(feature_files)
    test_files = [path for path in os.listdir("/mydata/features/test-set-real")]
    # test_index = random.sample(range(num_file), num_file % 100)
    # test_files = ["tc-0-tc-1-2290:0", "tc-0-tc-1-0:24300", "tc-0-tc-1-2290:2916", "tc-0-tc-1-22518:4053", "tc-0-tc-1-2243:488"]
    feature_set = ['sd_avg', 'iat_avg', 'size_avg', 'edc_avg']
    name_list = []
    feature_list = []
    test_set = [] 
    bucket_list = [10, 20, 50, 100, 500, 1000, 5000]
    
    # for i, set in enumerate([train_files, test_files]):
    for i, file in enumerate(train_files+test_files):
        feature = []
        name = file.split('.')[0]
        if not os.path.exists(os.path.join("/mydata/experts/", expert_0, name+'.pkl')):
            print(expert_0+'/'+name+'.pkl not exist')
            continue
        # name_list.append(name)
        if i < len(train_files):
            dir = "train-set"
        else:
            dir = "test-set-real"
            test_set.append(name)
                
    # for file in ["tc-0-tc-1-2290:0.pkl", "tc-0-tc-1-0:24300.pkl", "tc-0-tc-1-2290:2916.pkl", "tc-0-tc-1-22518:4053.pkl", "tc-0-tc-1-2243:488.pkl"]:
    # file = "tc-0-tc-1-138:958.pkl"
        
        features = pickle.load(open(os.path.join("/mydata/features/", dir, file), "rb"))
        for f in feature_set:
            v = features[f]
            if type(v) is dict or type(v) is defaultdict:
                values = [value for key,value in sorted(v.items())]   
                feature += values
            else:
                feature.append(v)
        # print(len(feature))
        feature_list.append(feature)
        
        assert len(feature) == 22

        for i, ele in enumerate(feature):
            if MAX_LIST[i] == 0 or MAX_LIST[i] < ele:
                MAX_LIST[i] = ele
            if MIN_LIST[i] == 0 or MIN_LIST[i] > ele:
                MIN_LIST[i] = ele
    
    for i, name in enumerate(name_list):
        
        if name in test_set:
            dir = "test-set-real"
        else:
            dir = "train-set"
        # print(i)
        # print(feature_list)
        # print(MIN_LIST)
        # print(MAX_LIST)
        # feature = [ (feature_list[i][j]- MIN_LIST[j])/(MAX_LIST[j]-MIN_LIST[j]) for j in range(len(MIN_LIST))]
        # feature = [ (feature_list[i][j]- MIN_LIST[j])/(MAX_LIST[j]-MIN_LIST[j]) for j in range(len(MIN_LIST))]
        try:
            e0_hits = pickle.load(open(os.path.join("/mydata/experts/", expert_0, name+'.pkl'), "rb"))
        except:
            print(expert_0+'/'+name+'.pkl load fails')
            continue
        try:
            e1_hits = pickle.load(open(os.path.join("/mydata/experts/", expert_1, name+'.pkl'), "rb"))
        except:
            print(expert_1+'/'+name+'.pkl load fails')
            continue
        hit_hit_prob = 0 # pi(e1_hit | e0_hit)
        e0_hit_count = 0
        e0_miss_count = 0
        hit_miss_prob = 0 # pi(e1_hit | e0_miss)
        
        assert (len(e0_hits) == len(e1_hits))
        
        count = 0
        # import pdb; pdb.set_trace()
        for e0, e1 in zip(e0_hits, e1_hits):
            if count < TRAINING_LEN:
                if e0 == 1:
                    e0_hit_count += 1
                    if e1 == 1:
                        hit_hit_prob += 1
                else:
                    e0_miss_count += 1
                    if e1 == 1:
                        hit_miss_prob += 1
            else:
                # if count == TRAINING_LEN:
                hit_hit_prob = hit_hit_prob/e0_hit_count
                hit_miss_prob = hit_miss_prob/e0_miss_count
                if dir == "test-set-real":
                    prediction_input.append([feature, e0_hit_count, e0_miss_count])
                    prediction_labels.append([hit_hit_prob, hit_miss_prob])
                else:
                    input.append(feature)
                    labels.append([hit_hit_prob, hit_miss_prob])
                break
                    # hit_hit_prob = e0_hit_count = e0_miss_count = hit_miss_prob = 0
                # if name not in test_files:
                #     break
                # if count >= TRAINING_LEN + PREDICTION_LEN:
                #     prediction_input.append([feature, e0_hit_count, e0_miss_count])
                #     hit_hit_prob = hit_hit_prob/e0_hit_count
                #     hit_miss_prob = hit_miss_prob/e0_miss_count
                #     prediction_labels.append([hit_hit_prob, hit_miss_prob])
                #     break
                # if e0 == 1:
                #     e0_hit_count += 1
                #     if e1 == 1:
                #         hit_hit_prob += 1
                # else:
                #     e0_miss_count += 1
                #     if e1 == 1:
                #         hit_miss_prob += 1
            count += 1
        if count > 0 and count % ROUND_LEN == 0:
            hit_hit_prob = hit_hit_prob/e0_hit_count
            hit_miss_prob = hit_miss_prob/e0_miss_count
            input = []
            input.extend(feature)
            input.extend(bucket_count)
            if name in test_set:
                prediction_input.append([input, e0_hit_count, e0_miss_count])
                prediction_labels.append([hit_hit_prob, hit_miss_prob])
            else:
                inputs.append(input)
                labels.append([hit_hit_prob, hit_miss_prob])
                
        if i % 100 == 0:
            pickle.dump(inputs, open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "traininput.pkl"), "wb"))
            pickle.dump(labels, open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "trainlabels.pkl"), "wb"))
            pickle.dump(prediction_input, open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "predinput.pkl"), "wb"))
            pickle.dump(prediction_labels, open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "predlabels.pkl"), "wb"))
        
    
    pickle.dump(inputs, open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "traininput.pkl"), "wb"))
    pickle.dump(labels, open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "trainlabels.pkl"), "wb"))
    pickle.dump(prediction_input, open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "predinput.pkl"), "wb"))
    pickle.dump(prediction_labels, open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "predlabels.pkl"), "wb"))
    return inputs, labels, prediction_input, prediction_labels


def test(device, epoch, model, _data, _label, batch_size):
    # import pdb; pdb.set_trace()
    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    print("=====Epoch {}=====".format(epoch))
    total_accuracy = 0
    
    # _data = torch.tensor(_data)
    # _label = torch.tensor(_label)
    # num_data, data_dim = _data.shape
    # # _, label_dim = label.shape
    
    # data_batched = _data.reshape(5, int(num_data / 5), data_dim)
    # label_batched = _label.reshape(5, int(num_data / 5))
    
    model.eval()
    for i_test in range(len(_data)): 
        data_i = _data[i_test]
        label_i = _label[i_test]
        # for d, l in zip(data_i, label_i):
        e0_hit_count = int(data_i[1])
        e0_miss_count = int(data_i[2])
        data_i = data_i[0]
        d = torch.tensor(data_i).to(device)
        l = torch.tensor(label_i).to(device)
        outputs = torch.sigmoid(model(d))
        hit_hit_prob = l.tolist()[0]
        hit_miss_prob = l.tolist()[1]
        pred_hit_hit_prob = outputs.tolist()[0]
        pred_hit_miss_prob = outputs.tolist()[1]
        e0_hitrate = e0_hit_count/(e0_hit_count + e0_miss_count) * 100
        real_e1_hitrate = (e0_hit_count * hit_hit_prob + e0_miss_count * hit_miss_prob) / (e0_hit_count + e0_miss_count) * 100
        pred_e1_hitrate = (e0_hit_count * pred_hit_hit_prob + e0_miss_count * pred_hit_miss_prob) / (e0_hit_count + e0_miss_count) * 100
        print("Trace {} results: ".format(i_test))
        print("e0 hit rate: {}".format(e0_hitrate))
        print("Real pi(e1_hit | e0_hit): {}".format(hit_hit_prob))
        print("Predicted pi(e1_hit | e0_hit): {}".format(pred_hit_hit_prob))
        print("Real pi(e1_hit | e0_miss): {}".format(hit_miss_prob))
        print("Predicted pi(e1_hit | e0_miss): {}".format(pred_hit_miss_prob))
        print("e1 real hit rate: {}".format(real_e1_hitrate))
        print("e1 predicted hit rate: {}".format(pred_e1_hitrate))
        sys.stdout.flush()

def main():
    hidden_size = int(sys.argv[1]) # hidden layer node number (2-9)
    expert_0 = sys.argv[2]
    expert_1 = sys.argv[3]
    device = sys.argv[4]
    # hidden_size = 5
    # expert_0 = "f4s50"
    # expert_1 = "f2s50"
    
    # Check Device configuration
    if torch.cuda.device_count() < 3:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device =  torch.device(device)
    
    # # Define Hyper-parameters 
    input_size = 22
    output_size = 2
    num_epochs = 1000
    # batch_size = 1000
    batch_size = 1
    learning_rate = 0.001


    # Fully connected neural network
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) 
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)  
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out
        
    if os.path.exists(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "model-h2.ckpt")):
        data = pickle.load(open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "traininput.pkl"), "rb"))
        label = pickle.load(open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "trainlabels.pkl"), "rb"))
        _data = pickle.load(open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "predinput.pkl"), "rb"))
        _label = pickle.load(open(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "predlabels.pkl"), "rb"))
    else:
        data, label, _data, _label = gen_data(expert_0, expert_1)
    data = torch.tensor(data)
    label = torch.tensor(label)
    # _data, _label = gen_predictdata(expert_0, expert_1, "tc-0-tc-1-2290:0")
    # Check if model exists
    if not os.path.exists(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "model-h"+str(hidden_size)+".ckpt")):
        # import pdb; pdb.set_trace()
        
        # data, label = gen_data(expert_0, expert_1)
        num_data, data_dim = data.shape
        # _, label_dim = label.shape
        
        num_batch = int(num_data / batch_size)
        data_batched = data.reshape(num_batch, batch_size, data_dim)
        label_batched = label.reshape(num_batch, batch_size, output_size)
        
        model = NeuralNet(input_size, hidden_size, output_size).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.MSELoss(reduction='sum')
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

        # Train the model
        # total_step = len(data)
        total_step = num_batch
        # start_t = time.time()
        for epoch in range(num_epochs):
            for i, (d, l) in enumerate(zip(data_batched, label_batched)):  
                # Move tensors to the configured device
                d = d.to(device)
                l = l.float().to(device)
                
                # Forward pass
                outputs = model(d)
                # import pdb; pdb.set_trace()
                # loss = criterion(outputs.squeeze(), l)
                loss = criterion(outputs, l)
                
                # Backprpagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    # end_t = time.time()
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                    # print('Training time: {}'.format(end_t-start_t))
                    # start_t = end_t
                    sys.stdout.flush()
            
            # Save the model checkpoint
            torch.save(model.state_dict(), os.path.join("/mydata/models/", expert_0+"-"+expert_1, "model-h"+str(hidden_size)+"-"+str(epoch)+".ckpt"))
            test(device, epoch, model, _data, _label, batch_size)
        
        torch.save(model.state_dict(), os.path.join("/mydata/models/", expert_0+"-"+expert_1, "model-h"+str(hidden_size)+".ckpt"))
        # print(model.state_dict())
    
    else:
        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load(os.path.join("/mydata/models/", expert_0+"-"+expert_1, "model-h"+str(hidden_size)+".ckpt")))
        test(device, 999, model, _data, _label, batch_size)
    

if __name__ == '__main__':
    main()