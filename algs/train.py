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
ROUND_LEN = 500000
MB = 1000000
MAX_LIST = [0] * 15
MIN_LIST = [0] * 15


def main():
    hidden_size = int(sys.argv[1]) # hidden layer node number (2-9)
    expert_0 = sys.argv[2]
    expert_1 = sys.argv[3]
    corr_dir = sys.argv[4]
    model_dir = sys.argv[5]
    
    # print("hidden_size: ", hidden_size)
    # print("expert_0: ", expert_0)
    # print("expert_1: ", expert_1)
    # print("corr_dir: ", corr_dir)
    # print("model_dir: ", model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # Define Hyper-parameters 
    input_size = 23
    output_size = 2
    num_epochs = 100
    # batch_size = 1000
    batch_size = 1
    learning_rate = 0.001
    data = []
    label = []


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
        
    for corr_file in os.listdir(corr_dir):
        print(corr_file)
        # load each pickle file and append to a list
        if corr_file.endswith("-input.pkl"):
            trace = '-'.join(corr_file.split("-")[:-1])
            print(trace)
            data_input = pickle.load(open(os.path.join(corr_dir, corr_file), "rb"))
            print(os.path.join(corr_dir, trace+"-labels.pkl"))
            label_input = pickle.load(open(os.path.join(corr_dir, trace+"-labels.pkl"), "rb"))
            for d,l in zip(data_input, label_input):
                data.append(d[0])
                label.append(l)
    pickle.dump(data, open(os.path.join(corr_dir, "traininput.pkl"), "wb"))
    pickle.dump(label, open(os.path.join(corr_dir, "trainlabels.pkl"), "wb"))

    data = torch.tensor(data)
    label = torch.tensor(label)
    data_u = data.mean(axis=0)
    data_sd = data.std(axis=0)
    pickle.dump(data_u, open(os.path.join(corr_dir, "traininput_u.pkl"), "wb"))
    pickle.dump(data_sd, open(os.path.join(corr_dir, "traininput_sd.pkl"), "wb"))

    if not os.path.exists(os.path.join(model_dir, "model-h"+str(hidden_size)+".ckpt")):
        # import pdb; pdb.set_trace()
        
        print(data.shape)
        [num_data, data_dim] = data.shape
        
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
                for j in range(data_dim):
                    d[0, j] = (d[0,j]-data_u[j])/data_sd[j]
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
            torch.save(model.state_dict(), os.path.join(model_dir, "model-h"+str(hidden_size)+"-"+str(epoch)+".ckpt"))
        
        torch.save(model.state_dict(), os.path.join(model_dir, "model-h"+str(hidden_size)+".ckpt"))
    

if __name__ == '__main__':
    main()