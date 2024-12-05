import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import torch.nn.utils.rnn 

plt.close('all')
device = "cuda" if torch.cuda.is_available() else "cpu"




## TODO añadir stride
## TODO revisar mismo delta en TODAS las series/arreglar producer
# TODO mover el astype a un lugar mas optimo
## TODO meter error de medicion
## TODO arreglar sliding window para ver si func modelo porque llevabamos a enteros
## TODO que hago con el volumen y el caudal de salida ver
## TODO Elegir Redim en base a limites reales
# TODO Preguntar que onda limites reales
# TODO Ver si curvas son reales




"""Utilidades"""





class TimeSeriesDataset(Dataset):
  def __init__(self, X, Y=None, sizes=None):
    self.X = X
    self.Y= Y
    self.sizes = sizes


  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):

    return self.X[ix], self.Y[ix], self.sizes[ix]

# TODO


class ModelLSTM(torch.nn.Module):
    
  def __init__(self, n_in=6, n_out=2, n_hidden=20, number_of_layers=10):
    super().__init__()
    self.lstm = torch.nn.LSTM(input_size=6, hidden_size=n_hidden, num_layers = number_of_layers, batch_first=True)
    self.fc = torch.nn.Linear(n_hidden, n_out)

  def forward(self, x):
    lstm_outs, (h_t, h_c) = self.lstm(x)   
    h_t=torch.transpose(h_t, 0, 1)
    
    x = self.fc(h_t)
    return x


def pad_and_pack(X, sizes):
    padded_X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    packed_X = torch.nn.utils.rnn.pack_padded_sequence(padded_X, sizes, batch_first=True, enforce_sorted=False)
    return packed_X


def fit(model, dataloader, epochs=20, learning_rate= 1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = torch.nn.MSELoss()
    bar = tqdm(range(1, epochs+1))
    for epoch in bar:
        
        model.train()
        train_loss = []
        for batch in dataloader['train']:
            
            X, Y, sizes = batch

            
            packed_X = pad_and_pack(X, sizes)
            Y=torch.stack(Y, dim=0)
            
            optimizer.zero_grad()
            
            Y_hat = model(packed_X)
            
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()
            print(loss.item())
            train_loss.append(loss.item())

            
        model.eval()
        eval_loss = []
        
        with torch.no_grad():
            for batch in dataloader['eval']:
                X, Y, sizes = batch

                
                packed_X = pad_and_pack(X, sizes)
                Y=torch.stack(Y, dim=0)
                Y_hat = model(packed_X)
                Y_hat = Y_hat.view(Y.shape)
                loss = criterion(Y_hat, Y)
                eval_loss.append(loss.item())
        bar.set_description(f"loss {np.mean(train_loss):.5f} val_loss {np.mean(eval_loss):.5f}")



def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
           preds = torch.tensor([]).to(device)
           for batch in dataloader:
               X, sizes = batch

               packed_X = pad_and_pack(X, sizes)
               
               pred = model(packed_X)
               preds = torch.cat([preds, pred])
           return preds 



def collate_fn_padd_and_pack(batch):
    ziped=batch
    x, y, size = zip(*ziped)
    
    return list(x), list(y), list(size)


"""#########################################################################################"""


from DefaultSeriesGenerator import *
from TensorGenerator import *
from AnalyticalRedimensionalizedSystem import *



a_list_of_systems = generate_original_series()


tensor_generator = TensorGenerator()

for i in range(len(a_list_of_systems)-1):
    redim_system = RedimensionalizedSystem(a_list_of_systems[i].series_dictionary) 
    redim_system.redimentionalizer_given_conditions_arrays([50, 25], [2, 3], [4, 1])
    tensor_generator.add_a_list(redim_system.list)
    


list_of_x, list_of_y, list_of_seq_len = tensor_generator.export_arrays(100, 10)



validation_index = int(len(list_of_x)*0.15)

X_traing = list_of_x[0:validation_index]
Y_traing = list_of_y[0:validation_index]
sizes_traing = list_of_seq_len[0:validation_index]

X_validg = list_of_x[validation_index:]
Y_validg = list_of_y[validation_index:]
sizes_validg = list_of_seq_len[validation_index:]


a_list_of_systems[-1]
redim_system = RedimensionalizedSystem(a_list_of_systems[i].series_dictionary) 
redim_system.redimentionalizer_given_conditions_arrays([50, 25, 30, 35], [2, 3, 2,4], [4, 1, 3,2])
X_testg, Y_testg, sizes_testg = tensor_generator.export_arrays(100, 10)




""" Preparo los datos """   


dataset = {
    'train': TimeSeriesDataset(X_traing, Y_traing, sizes_traing),
    'eval': TimeSeriesDataset(X_validg, Y_validg, sizes_validg),
    'test': TimeSeriesDataset(X_testg, Y_testg, sizes_testg)
}

dataloader = {
    'train': DataLoader(dataset['train'], shuffle=True, batch_size=64, collate_fn=collate_fn_padd_and_pack),
    'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64, collate_fn=collate_fn_padd_and_pack),
    'test': DataLoader(dataset['test'], shuffle=False, batch_size=64, collate_fn=collate_fn_padd_and_pack)
}



def collate_fn_padd_and_pack(batch):
    ziped=batch[0]
    x, y, size = zip(*ziped)
    
    return list(x), list(y), list(size)





# Creo y ajusto
modeloConError = ModelLSTM()
fit(modeloConError, dataloader, epochs=5, learning_rate= 1e-3)

# Grafico resultados


# = predict(modeloConError, dataloader = dataloader['test'])

for batch in dataloader['test']:
    x, y, z=batch
    



#y_pred = torch.squeeze(y_pred).numpy()
#Y_testg = np.squeeze(Y_testg)
#print('\n Mean squared error: ', mean_squared_error(Y_testg, y_pred))










