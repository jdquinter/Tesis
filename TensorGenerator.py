from AnalyticalRedimensionalizedSystem import *
import numpy as np
import random
import copy 
import torch



class TensorGenerator:
    
    def __init__(self):
        self.redim_dictionaries=[]
        self.field_names = ["outwards_concentration_series", "volume_series", "inwards_concentration_series", "outwards_flow_series", "inwards_flow_series", "time_series"]
        self.list_of_x = []
        self.list_of_y = []
        self.list_of_seq_len = []
        
    
    def add_a_list(self, list_of_redimed):
        self.redim_dictionaries.extend(list_of_redimed)
        
        
    def pass_to_lists(self, context_size, predict_size, as_arrays=False):
        
        for dictionary in self.redim_dictionaries:
            series_len = len(dictionary["outwards_concentration_series"])
            assert(series_len >= context_size + predict_size)
            start=0
            for positions_grabbed in range(1, series_len+1-predict_size):
                if(positions_grabbed>context_size):
                    start=start+1
                x = np.zeros(shape=(positions_grabbed-0, 6)).astype(np.float32)
                y= np.zeros(shape=(predict_size, 2)).astype(np.float32)
                for i in range(len(self.field_names)):
                    x[:,i]=dictionary[self.field_names[i]][0:positions_grabbed].astype(np.float32)
                    if(i<=1):
                        y[:,i]= dictionary[self.field_names[i]][positions_grabbed:positions_grabbed+10].astype(np.float32)
                        
                        
                
                
                #debug=True
                if(as_arrays):
                    self.list_of_x.append(x)
                    self.list_of_y.append(y)
                    self.list_of_seq_len.append(positions_grabbed)
                else:
                    self.list_of_x.append(torch.tensor(x))
                    self.list_of_y.append(torch.tensor(y))
                    self.list_of_seq_len.append(torch.tensor(positions_grabbed))
                    
                    
     

    
    def export_arrays(self, context_size, predict_size, as_arrays=False):
        self.pass_to_lists(context_size, predict_size, as_arrays)
        return(copy.deepcopy(self.list_of_x), copy.deepcopy(self.list_of_y), copy.deepcopy(self.list_of_seq_len))
        
        
        
        
        
        

if __name__ == '__main__':
    
    from DefaultSeriesGenerator import *
    from matplotlib import pyplot as plt
    
    a_physics_simulator = AnalyticalPhysicsSimulator()
    
    
    a_system = SystemSeries()
    a_system = SystemSeries()
    a_system.set_time_conditions(300, 1)
    a_system.set_starting_volume_conditions(30)
    a_system.set_steady_conditions("inwards_concentration_series", 2)
    a_system.set_steady_conditions("inwards_flow_series", 2)
    a_system.set_steady_conditions("outwards_flow_series", 1)
    a_system.simulate_physics(a_physics_simulator)

    
    redim_system = RedimensionalizedSystem(a_system.series_dictionary)    
    del a_system

    
    
    
    redim_system.redimentionalizer_given_conditions_arrays([50, 25], [2, 3], [4, 1])
    
    array_generator = TensorGenerator()
    
    array_generator.add_a_list(redim_system.list)
    
    
    list_of_x, list_of_y, list_of_seq_len = array_generator.export_arrays(100, 10)
    



            
            
    
