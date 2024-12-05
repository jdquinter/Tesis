#TODO name graphs


import numpy as np
import copy 

#time_series_string = "time_series"
#inwards_concentration_series_string= "inwards_concentration_series"
#outwards_concentration_series_string= "outwards_concentration_series"
#inwards_flow_series_string = "inwards_flow_series"
#outwards_flow_series_strin= "outwards_flow_series"
#volume_series_string = "volume_series"

class SystemSeries:
    
    def __init__(self):
        
        
        self.series_dictionary = {
           "time_series": None, 
           "inwards_concentration_series": None, 
           "outwards_concentration_series": None, 
           "inwards_flow_series": None, 
           "outwards_flow_series": None,  
           "volume_series": None
        }
        
        self.time_steps_amount = None
        
        
        
        
    
    

    def set_time_conditions(self, time_steps_amount, time_step_increment=1):
        
        self.series_dictionary["time_series"] = np.arange(stop = time_steps_amount * time_step_increment, step = time_step_increment, dtype = np.float64)
        self.time_steps_amount = time_steps_amount
        
        for name in self.series_dictionary.keys():
            if (name != "time_series"):
                self.series_dictionary[name]=np.zeros(time_steps_amount, dtype = np.float64)
        
        
        
 
    def set_starting_volume_conditions(self, starting_volume):
        self.series_dictionary["volume_series"][0] = starting_volume
    
    
    
    def set_steady_conditions(self, series_name, steady_condition, starting_time_step=0, ended_time_step = "True"):
        if ended_time_step == "True":
            ended_time_step = self.time_steps_amount
        assert(series_name not in ["time_series", "volume_series"])
        self.series_dictionary[series_name][starting_time_step:ended_time_step] = steady_condition
        
      
        
    def simulate_physics(self, physics_simulator):
        for name in self.series_dictionary.keys():
            assert (self.series_dictionary[name] is not None)
        
        physics_simulator.initialize_on(self)
        physics_simulator.simulate_physics()
    
    def import_from_dictionary(self, dictionary):
        self.series_dictionary = copy.deepcopy(dictionary)
        self.time_steps_amount =  len(self.series_dictionary["time_series"])



if __name__ == '__main__':
    
    from AnalyticalPhysicsSimulator import *
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

    
    plt.plot(a_system.series_dictionary["outwards_concentration_series"])
    plt.show()
    plt.plot(a_system.series_dictionary["volume_series"])
    plt.show()
        
        
        

    
        

        
        
        
#SystemSeries corresponde a un conjunto de series, todas correspondientes al mismo sistema.
#Las series relevantes son las siguientes:
# Serie de tiempo
# Serie de concentraciones de entrada
# Serie de concentraciones de salida
# Serie de caudales de entrada
# Serie de caudales de salida
# Serie de volumenes


