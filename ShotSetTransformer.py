#####################################################
# Training a SetTransformer on Silkeborg SHOT Data  #
# Author: Anika Kofod Petersen                      #
# Date: 12/6-24                                     #
# Status: Testing                                   #
#####################################################

# Define log writing function
def write_log(message, log_file):
    """ Write message to log file """
    with open(log_file,"a") as log:
        log.write(message+"\n")

# Define variables
out_path = "/path/to/setTransformer_test/"
data_path = "/path/to/SMD_SHOT/"
logfile = out_path+"log_file.txt"
dump_file = out_path+"dump_file.json"
batch_size = 128
lr = 1e-4
epochs = 15
early_stop_param = 3

#Import packages
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules_set_transformer import SAB, PMA, ISAB
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.preprocessing import normalize
import warnings
import lightning as L
import torchmetrics
warnings.filterwarnings("error")

# Check device and initialize log file
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
with open(logfile,"w+"):
    pass
write_log(f"CUDA available: {torch.cuda.is_available()}",logfile)
write_log(f"Device: {device}",logfile)
if device.type == 'cuda':
    write_log(f"Device name: {torch.cuda.get_device_name(0)}",logfile)

# Function for time formatting
def time_format(sec):
    """ Time formatting function """
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec) 

# Define SetTransformer architecture class
class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=16, dim_hidden=64, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        enc_dec = self.dec(self.enc(X))
        
        return torch.squeeze(enc_dec)

# Define function to check label for input data paths (only for Silkeborg Data)
def label_checker(path1,path2):
    """ Check label for two data paths """
    # Get ids
    id1 = path1.split("/")[-1].split("_")[0]
    id2 = path2.split("/")[-1].split("_")[0]

    # Check id match
    if id1 == id2:
        # Get jaws
        jaw1 = path1.split("/")[-1].split("Ma")[-1].split("_")[0]
        jaw2 = path2.split("/")[-1].split("Ma")[-1].split("_")[0]
        # Check jaw match
        if jaw1 == jaw2:
            # Get cut type
            cut1 = path1.split("/")[-1].split(".")[0][-2:]
            cut2 = path2.split("/")[-1].split(".")[0][-2:]
            # Check if match or mismatch
            if cut1 == cut2 or cut1 == "f0" or cut2 == "f0":
                return float(1)
            elif cut1 == "p1" and cut2 in ["p1","p3","p4"]:
                return float(1)
            elif cut1 == "p2" and cut2 in ["p2","p3","p4"]:
                return float(1)
            elif cut1 == "p3" and cut2 in ["p1","p2","p3"]:
                return float(1)
            elif cut1 == "p4" and cut2 in ["p1","p2","p4"]:
                return float(1)
            else:
                return float(0)
        else:
            return float(0)
    else:
        return float(0)

# Define dataset class
class SHOTDataset(Dataset):
    def __init__(self, in_path, data_pairs):
        super(SHOTDataset, self).__init__()
        self.in_path = in_path
        self.data_pairs = data_pairs

    def __getitem__(self, idx):

        #Get json dicts
        path1, path2 = os.path.join(self.in_path,self.data_pairs[idx][0]),os.path.join(self.in_path,self.data_pairs[idx][1])
        with open(path1,"rb") as d1:
            data1 = json.load(d1)
            
        with open(path2,"rb") as d2:
            data2 = json.load(d2)

        # Normalize shot
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        data1_n = torch.tensor(normalize(list(data1.values()),axis=1)).to(device)
        data2_n = torch.tensor(normalize(list(data2.values()),axis=1)).to(device)
        data_n = torch.cat((data1_n, data2_n), dim=0).to(dtype=torch.float32, device=device)
       
        # Get label
        label = torch.tensor(label_checker(path1,path2),).to(device)
        
        return data_n, label
            
    def  __len__(self):
        return len(self.data_pairs)

# Define custom collate function
def my_collate_fn(data):
    """ Custom collate function for SetTransformer training on Silkeborg Data """

    # maximum number of rows among the tensors
    max_rows = max(batch[0].shape[0] for batch in data)
  
    # Padding to make sizes compatible
    padded_data = [F.pad(batch[0], (0,0,0,max_rows - batch[0].size(0))) for batch in data]
  
    # Stack padded tensors
    x_data = torch.stack(padded_data, dim=0)

    # New data formatiing
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    out_data = (x_data, torch.tensor([data[batch][1] for batch in range(len(data))]).to(device))

    return out_data


# Initialize data sampling
write_log(f"DATASAMPLING",logfile)

# Random shuffle data
all_data_path_rand = os.listdir(data_path)[:100]   #REMOVE END FOR FULL DATA
random.seed(42)
random.shuffle(all_data_path_rand)

# Define negative subsampling frequency
probability = 1/(len(all_data_path_rand)/6)
probability = 0.03  # REMOVE THIS FOR FULL DATA

#Initialize counters
positives = 0
negatives = 0

# Set up data pairs, only keeping a subset of the negative samples
data_pairs = []
for i, data1 in enumerate(all_data_path_rand):
    if i != 0:
        subset = all_data_path_rand[i:]
    else:
        subset = all_data_path_rand
    for data2 in subset:
        if label_checker(data1,data2) != float(1):
            if random.random() < probability:
                data_pairs.append((data1,data2))
                negatives += 1
        else:
            positives += 1
            data_pairs.append((data1,data2))

write_log(f"Positives: {positives}  Negatives: {negatives}",logfile)
write_log(f"Ratios   : {round(positives/(positives+negatives),6)}    {round(negatives/(positives+negatives),6)}",logfile)


# Partition data into train, valid and test data
train = int(len(data_pairs)*0.6)
valid = int(len(data_pairs)*0.2)+train
test = len(data_pairs)
write_log(f"Train: {train}  Valid: {valid-train} Test: {test-valid}",logfile)
write_log(f"Train: {round((train/test)*100,2)}%  Valid: {round(((valid-train)/test)*100,2)}% Test: {round(((test-valid)/test)*100,2)}%",logfile)

# Prepare partitioned paths for loading
train_data_path = data_pairs[:train]
valid_data_path = data_pairs[train:valid]
test_data_path = data_pairs[valid:test]

# Calculating weights for BCE only based on training data (should be close to 1 for a uniform dataset)
lab_weight = []
zeros = 0
ones = 0
for path1,path2 in train_data_path:
    lab = label_checker(path1,path2)
    if lab == 0:
        zeros +=1
    elif lab == 1:
        ones += 1
    lab_weight.append(int(lab))
num_positives = sum(lab_weight)
num_negatives = len(lab_weight) - num_positives
pos_weight  = torch.Tensor([num_negatives / num_positives])

write_log(f"pos_weight: {pos_weight}",logfile)

# create DataLoader
trainLoader = DataLoader(SHOTDataset(data_path,train_data_path), batch_size=batch_size,
                        shuffle=True, collate_fn=my_collate_fn)
validLoader = DataLoader(SHOTDataset(data_path,valid_data_path), batch_size=batch_size,
                        shuffle=True, collate_fn=my_collate_fn)
testLoader = DataLoader(SHOTDataset(data_path,test_data_path), batch_size=batch_size,
                        shuffle=True, collate_fn=my_collate_fn)

# Update log file info
write_log(f"Number of Batches:",logfile)
write_log(f"trainLoader: {len(trainLoader)}",logfile)
write_log(f"validLoader: {len(validLoader)}",logfile)
write_log(f"testLoader: {len(testLoader)}",logfile)

# Define testing function to run for each epoch (both test and validation)
def test(model, test_loader):
    # initialize model and parameter saving
    model = model.cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    losses = []
    mcc = []
    model.eval() 
    batches = len(test_loader)

    # Turn off backtracking of gradients
    with torch.no_grad():

        # Iterate through batches
        for i_batch, sample_batched in enumerate(test_loader):

            # Perform testing
            x, y = sample_batched
            y_pred = model(x)
            sig = nn.Sigmoid()
            y_pred_log = sig(y_pred)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            write_log(f"     {i_batch+1}/{batches} Batch Loss: {loss.item()}",logfile)

            # Handle zero-division error and mark it as a complete random guess
            try:
                y_pred_log_ = torch.Tensor(y_pred_log.cpu().detach().numpy())
                mcc.append(MCC(y.cpu(), torch.round(y_pred_log_)))
            except RuntimeWarning:
                mcc.append(0)
                #breakpoint()

    # Get epoch mean and return
    losses = np.mean(losses)
    mcc = np.mean(mcc)
    return losses, mcc

# Initialize model and training
model = SetTransformer(352,1,1)
model= nn.DataParallel(model)  #FOR RUNNING ON MULTIPLE GPUS IF AVAILABLE
model.to(device)
scaler = torch.cuda.amp.GradScaler()
t0 = time.time()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()   #pos_weight=pos_weight

# Initialize metrics
train_losses = []
valid_losses = [100]
test_losses = []
test_mcc = []
valid_mcc = []
train_mcc = []
early_stop_counter = 0
epoch = 0
batches = len(trainLoader)

# Start training loop (early stop and max epochs)
write_log(f"TRAIN MODEL",logfile)
while epoch < epochs and early_stop_counter < early_stop_param:
    mcc = []
    epoch_loss = 0.0
    t1 = time.time()

    # No metrics for first epoch
    if epoch != 0:
        write_log(f"Epoch: {epoch}   Loss: {train_losses[-1]}",logfile)

    # Iterate through batches
    for i_batch, sample_batched in enumerate(trainLoader):
        optimizer.zero_grad()

        # Perform training
        x, y = sample_batched
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_pred = model(x)
            sig = nn.Sigmoid()
            y_pred_log = sig(y_pred)
            loss = criterion(y_pred, y)
            write_log(f"     {i_batch+1}/{batches} Batch Loss: {loss.item()}",logfile)

            # Handle zero-division error and mark it as a complete random guess
            try:
                y_pred_log_ = torch.Tensor(y_pred_log.cpu().detach().numpy())
                mcc.append(MCC(y.cpu(), torch.round(y_pred_log_)))
            except RuntimeWarning:
                mcc.append(0)

        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
            
        #optimizer.step()
        epoch_loss += loss.item() 

    #Log performance
    epoch_loss /= len(trainLoader)
    train_losses.append(epoch_loss)
    train_mcc.append(np.mean(mcc))

    #Validation and testing
    write_log(f"VALIDATE MODEL",logfile)
    v_loss, v_mcc = test(model, validLoader)
    write_log(f"TEST MODEL",logfile)
    t_loss, t_mcc = test(model, testLoader)

    #Check for early stopping
    if v_loss > valid_losses[-1]:
        early_stop_counter += 1
    else:
        early_stop_counter = 0

    #Log performance
    test_losses.append(t_loss)
    test_mcc.append(t_mcc)
    valid_losses.append(v_loss)
    valid_mcc.append(v_mcc)
    t2 = time.time()
    write_log(f"             Time for last epoch: {time_format(t2-t1)}",logfile)

    write_log(f"Early Stop Counter: {early_stop_counter}/{3}",logfile)
    epoch += 1
    
# Save model and end Log
valid_losses = valid_losses[1:]
t3 = time.time()
write_log(f"Time total: {time_format(t3-t0)}",logfile)
write_log(f"Done training.",logfile)
torch.save(model.state_dict(), out_path+"SetTransformer_stateDict.pt")
write_log(f"Model saved.",logfile)

#Save metrics for later use
metrics_dict = {}
metrics_dict["train_loss"] = train_losses
metrics_dict["valid_loss"] = valid_losses
metrics_dict["test_loss"] = test_losses
metrics_dict["train_mcc"] = train_mcc
metrics_dict["valid_mcc"] = valid_mcc
metrics_dict["test_mcc"] = test_mcc
with open(dump_file, "w") as df:
    json.dump(metrics_dict, df) 

#Plot MCC
plt.plot(test_mcc, label="Test MCC", color="lightsteelblue" )
plt.plot(valid_mcc, label="Validation MCC", color="cornflowerblue" )
plt.plot(train_mcc, label="Train MCC", color="navy" )
plt.legend(loc="upper left")
plt.xlabel("Epoch")
plt.xticks(range(len(train_losses)),range(1,len(train_losses)+1))
#plt.xticks(range(4,len(train_losses),10),range(5,len(train_losses)+1,10))
plt.ylabel("MCC")
plt.savefig(out_path+"MCC_plot.jpg")

# Loss plot
plt.plot(test_losses, label="Test Loss", color="lightsteelblue" )
plt.plot(valid_losses, label="Validation Loss", color="cornflowerblue" )
plt.plot(train_losses, label="Train Loss", color="navy")
plt.legend(loc="upper right")
plt.xlabel("Epoch")
plt.xticks(range(len(train_losses)),range(1,len(train_losses)+1))
#plt.xticks(range(4,len(train_losses),10),range(5,len(train_losses)+1,10))
plt.ylabel("BCE")
plt.savefig(out_path+"Loss_plot.jpg")


