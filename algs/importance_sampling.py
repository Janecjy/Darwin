import sys
import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

def gen_data(expert_0, expert_1):
    input = []
    labels = []
    
    feature_files = [path for path in os.listdir("../cache/output/features")]
    num_files = len(feature_files)
    train_num = 400
    feature_set = ['sd_avg', 'iat_avg', 'size_avg', 'edc_avg']
    name_list = []
    
    # for file in feature_files:
    file = "tc-0-tc-1-138:958.pkl"
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
    
    e0_hits = pickle.load(open("../cache/output/"+name+'/'+expert_0+'-hits.pkl', "rb"))
    e1_hits = pickle.load(open("../cache/output/"+name+'/'+expert_1+'-hits.pkl', "rb"))
    
    assert (len(e0_hits) == len(e1_hits))
    
    for e0, e1 in zip(e0_hits, e1_hits):
        x = feature[:]
        x.append(e0)
        input.append(x)
        y = [1, 0] if e1 == 1 else [0, 1]
        labels.append(y)
        
    return torch.tensor(input), torch.tensor(labels)

def main():
    hidden_size = int(sys.argv[1]) # hidden layer node number (2-9)
    expert_0 = sys.argv[2]
    expert_1 = sys.argv[3]
    
    # Check Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model exists
    if not os.path.exists('../cache/output/models/'+expert_0+'-'+expert_1+'-model.ckpt'):
    
        # Define Hyper-parameters 
        input_size = 23
        output_size = 2
        num_epochs = 5
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

        model = NeuralNet(input_size, hidden_size, output_size).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        data, label = gen_data(expert_0, expert_1)  

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
        torch.save(model.state_dict(), '../cache/output/models/'+expert_0+'-'+expert_1+'-model.ckpt')

    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images = images.reshape(-1, 28*28).to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    

if __name__ == '__main__':
    main()