import sys
import os
import pickle
import math
from random import choices
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np


TRAINING_LEN = 2000000
PREDICTION_LEN = 1000000
MB = 1000000
MAX_LIST = [0] * 22
MIN_LIST = [0] * 22

def gen_data(expert_0, expert_1):
    input = []
    labels = []
    
    feature_files = [path for path in os.listdir("../cache/output/features")]
    feature_set = ['sd_avg', 'iat_avg', 'size_avg', 'edc_avg']
    name_list = []
    feature_list = []
    
    # for file in feature_files:
    for file in ["tc-0-tc-1-2290:0.pkl", "tc-0-tc-1-0:24300.pkl", "tc-0-tc-1-2290:2916.pkl", "tc-0-tc-1-22518:4053", "tc-0-tc-1-2243:488.pkl"]:
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
            y = [1, 0] if e1 == 1 else [0, 1]
            if count < TRAINING_LEN:
                input.append(x)
                labels.append(y)
            else:
                break
            count += 1
        
    return torch.tensor(input), torch.tensor(labels)

def gen_predictdata(expert_0, expert_1, trace):
    prediction_input = []
    prediction_labels = []
    
    feature_files = [path for path in os.listdir("../cache/output/features")]
    num_files = len(feature_files)
    train_num = 400
    feature_set = ['sd_avg', 'iat_avg', 'size_avg', 'edc_avg']
    name_list = []
    feature_list = []
    
    # for file in feature_files:
    for file in ["tc-0-tc-1-2290:0.pkl", "tc-0-tc-1-0:24300.pkl", "tc-0-tc-1-2290:2916.pkl", "tc-0-tc-1-22518:4053", "tc-0-tc-1-2243:488.pkl"]:
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
    
    name = trace
    i = name_list.index(name)
    feature = [ (feature_list[i][j]- MIN_LIST[j])/(MAX_LIST[j]-MIN_LIST[j]) for j in range(len(MIN_LIST))]
    
    e0_hits = pickle.load(open("../cache/output/"+name+'/'+expert_0+'-hits.pkl', "rb"))
    e1_hits = pickle.load(open("../cache/output/"+name+'/'+expert_1+'-hits.pkl', "rb"))
    
    assert (len(e0_hits) == len(e1_hits))
    
    count = 0
    for e0, e1 in zip(e0_hits, e1_hits):
        x = feature[:]
        x.append(e0)
        y = [1, 0] if e1 == 1 else [0, 1]
        if count >= TRAINING_LEN:
            if count > TRAINING_LEN + PREDICTION_LEN:
                break
            prediction_input.append(x)
            prediction_labels.append(y)
        count += 1
    return torch.tensor(prediction_input), torch.tensor(prediction_labels)

def main():
    hidden_size = int(sys.argv[1]) # hidden layer node number (2-9)
    expert_0 = sys.argv[2]
    expert_1 = sys.argv[3]
    
    # Check Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # Define Hyper-parameters 
    input_size = 23
    output_size = 2
    num_epochs = 100
    batch_size = 100
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
    
    # Check if model exists
    if not os.path.exists('../cache/output/models/'+expert_0+'-'+expert_1+'-model-norm.ckpt'):
        
        data, label, _data, _label = gen_data(expert_0, expert_1)

        model = NeuralNet(input_size, hidden_size, output_size).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

        # Train the model
        total_step = len(data)
        for epoch in range(num_epochs):
            for i, (d, l) in enumerate(zip(data, label)):  
                # Move tensors to the configured device
                d = d.to(device)
                l = l.float().to(device)
                
                # Forward pass
                outputs = model(d)
                
                loss = criterion(outputs, l)
                
                # Backprpagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100000 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                    
        print(model.state_dict())
                    
        # Save the model checkpoint
        torch.save(model.state_dict(), '../cache/output/models/'+expert_0+'-'+expert_1+'-model-norm.ckpt')

    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    _data, _label = gen_predictdata(expert_0, expert_1, "tc-0-tc-1-2290:0")
     
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load('../cache/output/models/'+expert_0+'-'+expert_1+'-model-norm.ckpt'))
    model.eval()
    correct = 0
    prob_correct = 0
    total = 0
    true_hit = 0
    predicted_hit = 0
    prob_predicted_hit = 0
    for d, l in zip(_data, _label):
        # print(d, l)
        d = d.to(device)
        # l = l.float().to(device)
        outputs = model(d)
        # print(outputs)
        total += 1
        # if total == 10:
        #     break
        predicted = torch.argmax(outputs).item()
        real = torch.argmax(l).item()
        prob_list = outputs.tolist()
        prob = math.exp(prob_list[0])/(math.exp(prob_list[0])+math.exp(prob_list[1]))
        if predicted == real:
            correct += 1
        if predicted == 0:
            predicted_hit += 1
        if real == 0:
            true_hit += 1
        decision = choices([1, 0], [prob, 1-prob])
        if decision == 1:
            prob_predicted_hit += 1
        if decision == real:
            prob_correct += 1
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()

    print('Accuracy : {} %'.format(100 * correct / total))
    print('True Hit Rate : {} %'.format(100 * true_hit / total))
    print('Predicted Hit Rate : {} %'.format(100 * predicted_hit / total))
    print('Prob Accuracy : {} %'.format(100 * prob_correct / total))
    print('Prob Hit Rate : {} %'.format(100 * prob_predicted_hit / total))

    

if __name__ == '__main__':
    main()