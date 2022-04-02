import sys
import os
import pickle
import math
import time
from random import choices
from collections import defaultdict
from sympy import total_degree

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
    
    # feature_files = [path for path in os.listdir("../cache/output/features")]
    feature_set = ['sd_avg', 'iat_avg', 'size_avg', 'edc_avg']
    name_list = []
    feature_list = []
    
    # for file in feature_files:
    for file in ["tc-0-tc-1-2290:0.pkl", "tc-0-tc-1-0:24300.pkl", "tc-0-tc-1-2290:2916.pkl", "tc-0-tc-1-22518:4053.pkl", "tc-0-tc-1-2243:488.pkl"]:
    # file = "tc-0-tc-1-138:958.pkl"
        feature = []
        name = file.split('.')[0]
        name_list.append(name)
        features = pickle.load(open("../cache/output/features/"+file, "rb"))
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
        # print(i)
        # print(feature_list)
        # print(MIN_LIST)
        # print(MAX_LIST)
        feature = [ (feature_list[i][j]- MIN_LIST[j])/(MAX_LIST[j]-MIN_LIST[j]) for j in range(len(MIN_LIST))]
        
        e0_hits = pickle.load(open("../cache/output/"+name+'/'+expert_0+'-hits.pkl', "rb"))
        e1_hits = pickle.load(open("../cache/output/"+name+'/'+expert_1+'-hits.pkl', "rb"))
        
        assert (len(e0_hits) == len(e1_hits))
        
        count = 0
        for e0, e1 in zip(e0_hits, e1_hits):
            x = feature[:]
            x.append(e0)
            # y = [1, 0] if e1 == 1 else [0, 1]
            y = e1
            if count < TRAINING_LEN:
                input.append(x)
                labels.append(y)
            else:
                if count >= TRAINING_LEN + PREDICTION_LEN:
                    break
                prediction_input.append(x)
                prediction_labels.append(y)
            count += 1
    
    pickle.dump(input, open(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "traininput.pkl"), "wb"))
    pickle.dump(labels, open(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "trainlabels.pkl"), "wb"))
    pickle.dump(prediction_input, open(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "predinput.pkl"), "wb"))
    pickle.dump(prediction_labels, open(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "predlabels.pkl"), "wb"))
    return input, labels, prediction_input, prediction_labels


def test(device, epoch, model, _data, _label, batch_size):
    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    print("=====Epoch {}=====".format(epoch))
    total_accuracy = 0
    
    _data = torch.tensor(_data)
    _label = torch.tensor(_label)
    num_data, data_dim = _data.shape
    # _, label_dim = label.shape
    
    data_batched = _data.reshape(5, int(num_data / 5), data_dim)
    label_batched = _label.reshape(5, int(num_data / 5))
    
    model.eval()
    for i_test in range(5): 
        hit_prob_list = [-1, -1]
        data_i = data_batched[i_test]
        label_i = label_batched[i_test]
        correct = 0
        prob_correct = 0
        total = 0
        e0_hits = 0
        true_hit = 0
        predicted_hit = 0
        prob_predicted_hit = 0
        prob_known = False
        for d, l in zip(data_i, label_i):
            e0_hit = int(d.tolist()[22])
            if e0_hit == 1:
                e0_hits += 1
            if not prob_known:    
                if hit_prob_list[e0_hit] < 0:
                    d = d.to(device)
                    outputs = torch.sigmoid(model(d))
                    hit_prob = outputs.item()
                    hit_prob_list[e0_hit] = hit_prob
                if hit_prob_list[0] > 0 and hit_prob_list[1] > 0:
                    prob_known = True
            total += 1
            # if total == 10:
            #     break
            hit_prob = hit_prob_list[e0_hit]
            predicted = 1 if hit_prob > 0.5 else 0
            real = l.item()
            decision = choices([1, 0], [hit_prob, 1-hit_prob])[0]
            # import pdb; pdb.set_trace()
            
            if predicted == real:
                correct += 1
            if predicted == 1:
                predicted_hit += 1
            if real == 1:
                true_hit += 1
            
            if decision == 1:
                prob_predicted_hit += 1
            if decision == real:
                prob_correct += 1
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
        print("Trace {} results: ".format(i_test))
        print('Expert 0 Hit Rate : {} %'.format(100 * e0_hits / total))
        print('Expert 1 Hit Prediction Accuracy : {} %'.format(100 * correct / total))
        if (e0_hits > predicted_hit and e0_hits > true_hit) or (e0_hits <= predicted_hit and e0_hits <= true_hit):
            print('Experts Order Prediction is correct')
        else:
            print('Experts Order Prediction is wrong')
        print('True Expert 1 Hit Rate : {} %'.format(100 * true_hit / total))
        print('Predicted Expert 1 Hit Rate : {} %'.format(100 * predicted_hit / total))
        print('Probabilistically Predicted Expert 1 Accuracy : {} %'.format(100 * prob_correct / total))
        if (e0_hits > prob_predicted_hit and e0_hits > true_hit) or (e0_hits <= prob_predicted_hit and e0_hits <= true_hit):
            print('Experts Probabilistically Order Prediction is correct')
        else:
            print('Experts Probabilistically Order Prediction is wrong')
        print('Probabilistically Predicted Expert 1 Hit Rate : {} %'.format(100 * prob_predicted_hit / total))
        sys.stdout.flush()
        total_accuracy += 100 * correct / total

    print('Avg Accuracy : {} %'.format(total_accuracy / 5))

def main():
    hidden_size = int(sys.argv[1]) # hidden layer node number (2-9)
    expert_0 = sys.argv[2]
    expert_1 = sys.argv[3]
    # hidden_size = 5
    # expert_0 = "f4s50"
    # expert_1 = "f2s50"
    
    # Check Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # # Define Hyper-parameters 
    input_size = 23
    output_size = 1
    num_epochs = 50
    batch_size = 1000
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
        
    if os.path.exists(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "traininput.pkl")):
        data = pickle.load(open(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "traininput.pkl"), "rb"))
        label = pickle.load(open(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "trainlabels.pkl"), "rb"))
        _data = pickle.load(open(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "predinput.pkl"), "rb"))
        _label = pickle.load(open(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "predlabels.pkl"), "rb"))
    else:
        data, label, _data, _label = gen_data(expert_0, expert_1)
    data = torch.tensor(data)
    label = torch.tensor(label)
    # _data, _label = gen_predictdata(expert_0, expert_1, "tc-0-tc-1-2290:0")
    # Check if model exists
    if not os.path.exists(os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "model.ckpt")):
        
        # data, label = gen_data(expert_0, expert_1)
        num_data, data_dim = data.shape
        # _, label_dim = label.shape
        
        num_batch = int(num_data / batch_size)
        data_batched = data.reshape(num_batch, batch_size, data_dim)
        label_batched = label.reshape(num_batch, batch_size)
        
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
                loss = criterion(outputs.squeeze(), l)
                
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
            torch.save(model.state_dict(), os.path.join("../cache/output/models/", expert_0+"-"+expert_1, "model-"+str(epoch)+".ckpt"))
            test(device, epoch, model, _data, _label, batch_size)
                    
        # print(model.state_dict())
    
    else:
        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load('../cache/output/models/'+expert_0+'-'+expert_1+'-model-norm.ckpt'))
        test(device, 0, model, _data, _label, batch_size)
    

if __name__ == '__main__':
    main()