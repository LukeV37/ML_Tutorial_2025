# Import Tensor Libs
import numpy as np
import awkward as ak

# Import ML Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import Plotting Libs
import matplotlib.pyplot as plt

# Check if GPU is available, if not use cpu
print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ParticleDataset(Dataset):
    def __init__(self, data, labels, device):
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        event = self.data[idx]
        label = self.labels[idx]
        return event, label

train_dataset = torch.load("../datasets/train_dataset.pt", weights_only=False, map_location=device)
val_dataset = torch.load("../datasets/val_dataset.pt", weights_only=False, map_location=device)
test_dataset = torch.load("../datasets/test_dataset.pt", weights_only=False, map_location=device)

train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

for inputs, labels in train_dataloader:
    print("Batch shape:", inputs.shape)
    print("Labels shape:", labels.shape)
    padding_particle_cutoff=inputs.shape[1]
    break

# Define the Model
class MultiLayerPerceptron(nn.Module):
    '''
    A DL model with customizable layers and nodes. 
    '''
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        hidden_layers = [hidden_dim]*num_layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers[:-1]:
            x = F.gelu(layer(x))
        x = F.sigmoid(self.layers[-1](x))
        return x

# Define a class that inherits from torch.nn.Module
class DeepSets(nn.Module):
    '''
    A DeepSets model that performs graph level classification on a dense graph. 
    '''
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DeepSets, self).__init__()
        self.init = nn.Linear(in_dim, hidden_dim)
        self.Messages = nn.Linear(hidden_dim,hidden_dim)
        self.PostProcess = nn.Linear(hidden_dim,hidden_dim)
        self.Classification = nn.Linear(hidden_dim,out_dim)
    def forward(self, data):
        track_embedding = F.gelu(self.init(data))
        messages = F.gelu(self.Messages(track_embedding))
        aggregated_message = torch.sum(messages,dim=1)
        event_embedding = F.gelu(self.PostProcess(aggregated_message))
        output = F.sigmoid(self.Classification(event_embedding))
        return output

# Define a class that inherits from torch.nn.Module
class TransformerEncoder(nn.Module):
    '''
    An Attention model that performs set level classification on a set. 
    '''
    def __init__(self, in_dim, hidden_dim, num_encoders, out_dim):
        super(TransformerEncoder, self).__init__()
        self.init = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, dim_feedforward=hidden_dim,dropout=0, batch_first=True)        # Define linear transformation 1
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.Classification = nn.Linear(hidden_dim,out_dim)
    def forward(self, data):
        embedding = F.gelu(self.init(data))
        embedding = F.gelu(self.transformer_encoder(embedding))
        embedding = torch.mean(embedding,dim=1)
        output = F.sigmoid(self.Classification(embedding))
        return output

# Define the training loop
def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, lr_step, epochs=20):

    history = {'train_loss':[],'test_loss':[]}     # Define history dictionary

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)
    # Loop through epoches
    for e in range(epochs):
        for X_train, y_train in train_dataloader:
            # Train Model
            model.train()                        # Switch model to training mode
            optimizer.zero_grad()                # Reset the optimizers gradients
            y_pred = model(X_train)             # Get the model prediction
            loss = loss_fn(y_pred, y_train)     # Evaluate loss function
            loss.backward()                      # Backward propogation
            optimizer.step()                     # Gradient Descent
        for X_val, y_val in val_dataloader:
            # Validate Model
            model.eval()
            y_pred = model(X_val)    # Get model output on test data
            test_loss = loss_fn(y_pred,y_val)   # Evaluate loss on test preditions
        
        history['train_loss'].append(loss.detach().cpu().numpy())        # Append train loss to history (detach and convert to numpy array)
        history['test_loss'].append(test_loss.detach().cpu().numpy())    # Append test loss to history (detach and convert to numpy array)
        if (e+1)%1==0:
            print('Epoch:',e+1,'\tTrain Loss:',round(float(loss),4),'\tTest Loss:',round(float(test_loss),4))
        scheduler.step()
        if (e+1)%lr_step==0:
            print("\tReducing Learning Rate!")
    return history

# Initialize model
MLP = MultiLayerPerceptron(in_dim=3*padding_particle_cutoff,hidden_dim=8, num_layers=2, out_dim=1)    # Declare model using NeuralNet Class
MLP.to(device)                                       # Put model on device (cpu or gpu)
print(MLP)                                           # Print layers in model

# Calculate and print trainable parameters
pytorch_total_params = sum(p.numel() for p in MLP.parameters() if p.requires_grad)
print("Trainable Parameters: ", pytorch_total_params,"\n")

# Declare optimizer and loss function
optimizer = optim.AdamW(MLP.parameters(), lr=1e-6)  # Adam = Adaptive Moment Estimation, lr = learning rate
loss_fn = nn.BCELoss()                                 # BCE = Binary Cross Entropy, used for binary classification

#Train Model
history = train(MLP, optimizer, loss_fn, train_dataloader, val_dataloader, lr_step=15, epochs=30)                                             # Train the model!

# Plot Training History
plt.figure()
plt.plot(history['train_loss'],label='train')
plt.plot(history['test_loss'],label='test')
plt.title("Loss")
plt.legend()
plt.yscale('log')
plt.savefig("MLP_loss.png")
plt.show()

torch.save(MLP, "MLP.pt")

# Initialize model
GNN = DeepSets(in_dim=3,hidden_dim=16,out_dim=1)    # Declare model using NeuralNet Class
GNN.to(device)                                       # Put model on device (cpu or gpu)
print(GNN)                                           # Print layers in model

# Calculate and print trainable parameters
pytorch_total_params = sum(p.numel() for p in GNN.parameters() if p.requires_grad)
print("Trainable Parameters: ", pytorch_total_params,"\n")

# Declare optimizer and loss function
optimizer = optim.AdamW(GNN.parameters(), lr=1e-6)  # Adam = Adaptive Moment Estimation, lr = learning rate
loss_fn = nn.BCELoss()                                 # BCE = Binary Cross Entropy, used for binary classification

#Train Model
history = train(GNN, optimizer, loss_fn, train_dataloader, val_dataloader, lr_step=15, epochs=30)                                             # Train the model!

# Plot Training History
plt.figure()
plt.plot(history['train_loss'],label='train')
plt.plot(history['test_loss'],label='test')
plt.title("Loss")
plt.legend()
plt.yscale('log')
plt.savefig("GNN_loss.png")
plt.show()

torch.save(GNN, "GNN.pt")

# Initialize model
Transformer = TransformerEncoder(in_dim=3,hidden_dim=16,num_encoders=2,out_dim=1)    # Declare model using NeuralNet Class
Transformer.to(device)                                       # Put model on device (cpu or gpu)
print(Transformer)                                           # Print layers in model

# Calculate and print trainable parameters
pytorch_total_params = sum(p.numel() for p in Transformer.parameters() if p.requires_grad)
print("Trainable Parameters: ", pytorch_total_params,"\n")

# Declare optimizer and loss function
optimizer = optim.AdamW(Transformer.parameters(), lr=1e-5)  # Adam = Adaptive Moment Estimation, lr = learning rate
loss_fn = nn.BCELoss()                                 # BCE = Binary Cross Entropy, used for binary classification

#Train Model
history = train(Transformer, optimizer, loss_fn, train_dataloader, val_dataloader, lr_step=15, epochs=30)                                             # Train the model!

# Plot Training History
plt.figure()
plt.plot(history['train_loss'],label='train')
plt.plot(history['test_loss'],label='test')
plt.title("Loss")
plt.legend()
plt.yscale('log')
plt.savefig("Transformer_loss.png")
plt.show()

torch.save(Transformer, "Transformer.pt")

MLP = torch.load("MLP.pt", weights_only=False, map_location=device)
GNN = torch.load("GNN.pt", weights_only=False, map_location=device)
Transformer = torch.load("Transformer.pt", weights_only=False, map_location=device)

# Define traditional ROC curve
def roc(y_pred,y_true):    
    sig_eff = []
    bkg_eff = []
    
    sig = y_true==1
    bkg = y_true==0
    
    thresholds = np.linspace(0,1,100)
    
    # Iterate over thresholds and calculate sig and bkg efficiency
    for threshold in thresholds:
        sig_eff.append(((y_pred[sig] > threshold).sum() / y_true[sig].shape[0]))   # Sum over sig predictions > threshold and divide by total number of true sig instances 
        bkg_eff.append(((y_pred[bkg] < threshold).sum()  / y_true[bkg].shape[0]))  # Sum over bkg predictions < threshold and divide by total number of true bkg instances 
        
    return np.array(sig_eff), np.array(bkg_eff), thresholds

# Define ATLAS Style ROC curve
def ATLAS_roc(y_pred,y_true):
    sig_eff = []
    bkg_eff = []
    
    sig = y_true==1
    bkg = y_true==0
    
    thresholds = np.linspace(0,1,1000)
    
    for threshold in thresholds:
        sig_eff.append(((y_pred[sig] > threshold).sum() / y_true[sig].shape[0]))
        bkg_eff.append(1-((y_pred[bkg] < threshold).sum()  / y_true[bkg].shape[0]))
        
    bkg_rej = [1/x for x in bkg_eff]  # ATLAS inverts bkg eff and uses bkg rejection instead
    return np.array(sig_eff), np.array(bkg_rej), thresholds

def eval_model(model, loss_fn, test_dataloader):
    # Get Models predictions
    prediction = []
    truth = []
    test_loss = []
    for X_test, y_test in test_dataloader:
        model.eval()
        y_pred = model(X_test)    # Get model output on test data
        loss = loss_fn(y_pred,y_test)
        test_loss.append(loss.detach().cpu().numpy())   # Evaluate loss on test preditions
        prediction.append(y_pred.detach().cpu().numpy())
        truth.append(y_test.detach().cpu().numpy())

    prediction = np.array(ak.ravel(prediction))
    truth = np.array(ak.ravel(truth))
    
    # Find indices of sig and bkg labels
    sig = np.where(truth==1)
    bkg = np.where(truth==0)
    
    eff_sig, eff_bkg, thresh = roc(prediction,truth)
    
    WPs = [25,50,75]
    cuts = []
    for WP in WPs:
        mask = eff_sig>(WP/100)
        idx = len(eff_sig[mask])-1
        cut = thresh[idx]
        cuts.append(cut)

    # Plot Model Predictions
    plt.title(model.__class__.__name__+" Predictions")
    plt.hist(prediction[sig],histtype='step',color='r',label="sig",bins=40)
    plt.hist(prediction[bkg],histtype='step',color='b',label="bkg",bins=40)
    colors=['c','m','y']
    labels=[str(x)+"% WP" for x in WPs]
    for i, cut in enumerate(cuts):
        plt.axvline(cut,linestyle='--',color=colors[i%3],label=labels[i])
    plt.xlabel("Model Score")
    plt.ylabel("Events")
    #plt.yscale('log')
    plt.legend()
    plt.show()
    
    return prediction, truth

loss_fn = nn.BCELoss()
MLP_pred, MLP_true = eval_model(MLP, loss_fn, test_dataloader)
GNN_pred, GNN_true = eval_model(GNN, loss_fn, test_dataloader)
Trans_pred, Trans_true = eval_model(Transformer, loss_fn, test_dataloader)

# Plot Tradiation ROC Curve
MLP_eff_sig, MLP_eff_bkg, MLP_thresh = roc(MLP_pred,MLP_true)
GNN_eff_sig, GNN_eff_bkg, GNN_thresh = roc(GNN_pred,GNN_true)
Trans_eff_sig, Trans_eff_bkg, Trans_thresh = roc(Trans_pred,Trans_true)

plt.figure()
plt.title("ROC Curve")
plt.plot(MLP_eff_sig,MLP_eff_bkg,color='r',label="MLP")
plt.plot(GNN_eff_sig,GNN_eff_bkg,color='g',label="GNN")
plt.plot(Trans_eff_sig,Trans_eff_bkg,color='b',label="Transformer")
plt.plot([1,0],'--',color='k',label="Random Model")
plt.xlabel("Signal Efficiency")
plt.ylabel("Background Efficiency")
plt.legend()
plt.savefig("Classic_ROC.png")
plt.show()

# Plot ATLAS Style ROC Curve
MLP_eff_sig, MLP_eff_bkg, MLP_thresh = ATLAS_roc(MLP_pred,MLP_true)
GNN_eff_sig, GNN_eff_bkg, GNN_thresh = ATLAS_roc(GNN_pred,GNN_true)
Trans_eff_sig, Trans_eff_bkg, Trans_thresh = ATLAS_roc(Trans_pred,Trans_true)

plt.figure()
plt.title("ATLAS ROC Curve")
plt.plot(MLP_eff_sig,MLP_eff_bkg,color='r',label="MLP")
plt.plot(GNN_eff_sig,GNN_eff_bkg,color='g',label="GNN")
plt.plot(Trans_eff_sig,Trans_eff_bkg,color='b',label="Transformer")
plt.xlabel("Signal Efficiency")
plt.ylabel("Background Rejection")
plt.yscale('log')
plt.grid(True,which='both')
plt.xlim([0, 1])
plt.legend()
plt.savefig("ATLAS_ROC.png")
plt.show()

plt.figure()
plt.title("ATLAS ROC Curve")
plt.plot(MLP_eff_sig,MLP_eff_bkg,color='r',label="MLP")
plt.plot(GNN_eff_sig,GNN_eff_bkg,color='g',label="GNN")
plt.plot(Trans_eff_sig,Trans_eff_bkg,color='b',label="Transformer")
plt.xlabel("Signal Efficiency")
plt.ylabel("Background Rejection")
plt.yscale('log')
plt.grid(True,which='both')
plt.xlim([0, 0.2])
plt.legend()
plt.savefig("ATLAS_ROC_Low_sigEff.png")
plt.show()

plt.figure()
plt.title("ATLAS ROC Curve")
plt.plot(MLP_eff_sig,MLP_eff_bkg,color='r',label="MLP")
plt.plot(GNN_eff_sig,GNN_eff_bkg,color='g',label="GNN")
plt.plot(Trans_eff_sig,Trans_eff_bkg,color='b',label="Transformer")
plt.xlabel("Signal Efficiency")
plt.ylabel("Background Rejection")
plt.yscale('log')
plt.grid(True,which='both')
plt.xlim([0.7, 1])
plt.ylim([0, 18])
plt.legend()
plt.savefig("ATLAS_ROC_High_sigEff.png")
plt.show()
