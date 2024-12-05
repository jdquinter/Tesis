import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import pandas as pd
from Generador5 import * 

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



def singleStepSampler(tupla_listas_series, only_predict_dim, predict_dim, window, predict_window):
    # Requiere que el predict sean los primeros n dims
    
    cantidad_samples = 0
    dim_x = len(tupla_listas_series)-only_predict_dim
    
    for i in range(len(tupla_listas_series[0])):
        tamaño_serie = tupla_listas_series[0][i].shape[0]
        
        if(window > tamaño_serie): raise Exception(f" Series {i} with lenght {tamaño_serie} shorter than windo size {window}")
        
        cantidad_samples = cantidad_samples + tamaño_serie - window + 1
        
    X = np.zeros((cantidad_samples, dim_x, window - predict_window))
    Y = np.zeros((cantidad_samples, predict_dim, predict_window))
    
    sample_number=0
    for serie_number in range(len(tupla_listas_series[0])):       
        i=0
        while(i + window <= len(tupla_listas_series[0][serie_number])):
                       
            for dim in range(len(tupla_listas_series)):
                if(dim<predict_dim):
                    Y[sample_number, dim, :] = tupla_listas_series[dim][serie_number][i + window - predict_window: i + window]
   
                    
                if(dim>=only_predict_dim):
                    X[sample_number, dim, :] = tupla_listas_series[dim][serie_number][i:i + window - predict_window]
            
            i = i + 1
            sample_number = sample_number + 1
            
    return([X.astype(np.float32), Y.astype(np.float32)])
    



def permuteXY(XY, wanted_seed):
    rng = np.random.default_rng(seed=wanted_seed)
    rng.shuffle(XY[0], axis=0)
    rng = np.random.default_rng(seed=wanted_seed)
    rng.shuffle(XY[1], axis=0)



class TimeSeriesDataset(Dataset):
  def __init__(self, X, Y=None, train=True):
    self.X = X
    self.Y= Y
    self.train = train

  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):
    if self.train:
      return torch.tensor(self.X[ix], requires_grad=True), torch.tensor(self.Y[ix], requires_grad=True)
    return torch.tensor(self.X[ix], requires_grad=True)




class ModelLSTM(torch.nn.Module):
    
  def __init__(self, n_in=40, n_out=10, n_hidden=20, number_of_layers=4):
    super().__init__()
    self.lstm = torch.nn.LSTM(input_size=n_in, hidden_size=n_hidden, num_layers = number_of_layers, batch_first=True)
    self.fc = torch.nn.Linear(n_hidden, n_out)

  def forward(self, x):
    x, h = self.lstm(x) 
    x = self.fc(x[:,-1])
    return x




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
            X, Y = batch
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_hat = model(X)
            Y_hat = Y_hat.view(Y.shape)
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            
        model.eval()
        eval_loss = []
        
        with torch.no_grad():
            for batch in dataloader['eval']:
                X, Y = batch
                X, Y = X.to(device), Y.to(device)
                Y_hat = model(X)
                Y_hat = Y_hat.view(Y.shape)
                loss = criterion(Y_hat, Y)
                eval_loss.append(loss.item())
        bar.set_description(f"loss {np.mean(train_loss):.5f} val_loss {np.mean(eval_loss):.5f}")



def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
           preds = torch.tensor([]).to(device)
           for batch in dataloader:
               X = batch
               X = X.to(device)
               pred = model(X)
               preds = torch.cat([preds, pred])
           return preds 



def autoRegression(model, curva, predice, usa):
    model.eval()
    cant_predicciones = int((curva[0].size - usa) / 10)   
    pred=None
    
    regressed = curva[0][0:usa]
    
    for i in range(cant_predicciones):
        
        X=np.zeros((len(curva), usa))
        for dim in range(len(curva)):
            if(dim==0):
                X[dim] = regressed[-40: regressed.size]
            else:    
                X[dim] = curva[dim][i*10:40+i*10]
                
        X=torch.tensor(X.astype(np.float32)).view((1, 4, 40))
        
        pred = model(X).view((predice)).detach().numpy()

        regressed = np.concatenate((regressed, pred))
    return regressed





"""Carga de datos"""



predice = 10
n_steps = 40

data = producirDatos(cant_condiciones= 15, min_conc_entrada= 2, max_conc_entrada= 3, min_caud_entrada=5, max_caud_entrada=7, interp=True)


sampleado_de_datos = singleStepSampler(data, 0, 1, predice + n_steps, predice)



"""Separacion de Datos"""



permuteXY(sampleado_de_datos, 12)


#sampleado_de_datos=[sampleado_de_datos[0][0:2000], sampleado_de_datos[1][0:2000]]
sampleado_de_datos[0] = (sampleado_de_datos[0] + np.random.normal(0,.01, np.shape(sampleado_de_datos[0]))).astype(np.float32)

####


validation_index = int(len(sampleado_de_datos[0])*0.15)
testing_percent = int(len(sampleado_de_datos[0])*0.30) 







X_validg = sampleado_de_datos[0][0:validation_index]
Y_validg = sampleado_de_datos[1][0:validation_index]

X_testg = sampleado_de_datos[0][validation_index:testing_percent]
Y_testg = sampleado_de_datos[1][validation_index:testing_percent]

X_traing = sampleado_de_datos[0][testing_percent:]
Y_traing = sampleado_de_datos[1][testing_percent:]



""" Preparo los datos """


dataset = {
    'train': TimeSeriesDataset(X_traing, Y_traing),
    'eval': TimeSeriesDataset(X_validg, Y_validg),
    'test': TimeSeriesDataset(X_testg, Y_testg, train=False)
}

dataloader = {
    'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
    'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
    'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}





# Creo y ajusto
modeloConError = ModelLSTM(n_in=n_steps, n_out=predice, n_hidden=20, number_of_layers=10)
fit(modeloConError, dataloader, epochs=20, learning_rate= 1e-3)

# Grafico resultados


y_pred = predict(modeloConError, dataloader = dataloader['test'])



y_pred = torch.squeeze(y_pred).numpy()
Y_testg = np.squeeze(Y_testg)
print('\n Mean squared error: ', mean_squared_error(Y_testg, y_pred))




for i in range(30): 

    indice= 20*i
    
    regressed = autoRegression(modeloConError, [data[0][indice], data[1][indice], data[2][indice], data[3][indice]], 10, 40)
    
    
    plt.plot(data[0][indice])
    plt.plot(regressed)
    plt.show()




# TODO Error
# TODO Maskear
# TODO Regim, Solo dos curvas
# TODO Cambiar test


# Predecir 15 mins
