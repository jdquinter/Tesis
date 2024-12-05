#TODO compare function
#TODO name graphs

import copy

class RedimensionalizedSystem:
    def __init__(self, dictionary):
        self.original_series_dictionary = copy.deepcopy(dictionary)
        self.adimentionalized_series_dictionary = copy.deepcopy(dictionary)
        self.adimentionalize()
        self.list = []


    def adimentionalize(self):     
        
        self.adimentionalized_series_dictionary["volume_series"] = self.original_series_dictionary["volume_series"] /self.original_series_dictionary["volume_series"][0]
        self.adimentionalized_series_dictionary["time_series"] = self.original_series_dictionary["time_series"] * self.original_series_dictionary["outwards_flow_series"][0]/ self.original_series_dictionary["volume_series"][0]
        self.adimentionalized_series_dictionary["outwards_concentration_series"] = self.original_series_dictionary["outwards_concentration_series"] /self.original_series_dictionary["inwards_concentration_series"][0]
        
        self.adimentionalized_series_dictionary["inwards_concentration_series"] = self.original_series_dictionary["inwards_concentration_series"] /self.original_series_dictionary["inwards_concentration_series"][0]
        self.adimentionalized_series_dictionary["outwards_flow_series"] = self.original_series_dictionary["outwards_flow_series"] / self.original_series_dictionary["outwards_flow_series"][0]


    def redimentionalize(self, starting_volume, starting_outwards_flow, starting_inwards_concentration):
        redim_system_dictionary = copy.deepcopy(self.adimentionalized_series_dictionary)
        redim_system_dictionary["volume_series"] = redim_system_dictionary["volume_series"] * starting_volume
        redim_system_dictionary["time_series"] = redim_system_dictionary["time_series"] * starting_volume/ starting_outwards_flow
        redim_system_dictionary["outwards_concentration_series"] = redim_system_dictionary["outwards_concentration_series"] * starting_inwards_concentration
        
        redim_system_dictionary["inwards_concentration_series"] = redim_system_dictionary["inwards_concentration_series"] * starting_inwards_concentration
        redim_system_dictionary["outwards_flow_series"] = redim_system_dictionary["outwards_flow_series"] / redim_system_dictionary["outwards_flow_series"][0]
        
        
        self.list.append(redim_system_dictionary)
    
    
    def redimentionalizer_given_conditions_arrays(self, list_of_starting_volumes, list_of_starting_outwards_flow, list_of_starting_inwards_concentrations):
        amount_of_conditions =len(list_of_starting_volumes)
        for starting_volume in list_of_starting_volumes:
            for starting_outwards_flow in list_of_starting_outwards_flow:
                for starting_inwards_concentration in list_of_starting_inwards_concentrations:
                    self.redimentionalize(starting_volume, starting_outwards_flow, starting_inwards_concentration)
    
        
        
        
        
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
    plt.plot(redim_system.adimentionalized_series_dictionary["outwards_concentration_series"])
    plt.show()
    
    
    
    redim_system.redimentionalizer_given_conditions_arrays([50, 25], [2, 3], [4, 1])
    
    plt.plot(redim_system.list[0]["outwards_concentration_series"])
    plt.show()
    
    
    

    