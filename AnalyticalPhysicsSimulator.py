import numpy as np


class AnalyticalPhysicsSimulator:
    
    def __init__(self):
        
        self.system = None
        
        self.condition_changes_time_steps = None

    def initialize_on(self, a_system_series):
        self.system = a_system_series
        self.condition_changes_time_steps = [False] * self.system.time_steps_amount
        
        for name in self.system.series_dictionary.keys():
            if name not in ["time_series", "volume_series"]:
                serie=self.system.series_dictionary[name]
                
                previous_value=serie[0]
                for step in range(self.system.time_steps_amount):
                    if (previous_value != serie[step]):
                        self.condition_changes_time_steps[step+1] = True
                        previous_value=serie[step]
    
                        
    

    def compute_volume_increment(self, time_difference, previous_volume, previous_inwards_flow, previous_outwards_flow):
        return previous_volume + (previous_inwards_flow - previous_outwards_flow) * time_difference
        
    
    def compute_volume(self):
        for step in range(1, self.system.time_steps_amount):
            self.system.series_dictionary["volume_series"][step] = self.compute_volume_increment(self.system.series_dictionary["time_series"][1], self.system.series_dictionary["volume_series"][step-1], self.system.series_dictionary["inwards_flow_series"][step-1], self.system.series_dictionary["outwards_flow_series"][step-1])
    
    def compute_concentrations_for_conditions1(self, time_in_the_series, inwards_flow, outwards_flow, inwards_concentration, starting_concentration, starting_volume):   
            
        if inwards_flow == outwards_flow:     
            inwards_flow = inwards_flow + 0.00001
            
        outwards_flow_diff = outwards_flow - inwards_flow
        
        correction_base= 1  +  (time_in_the_series*outwards_flow_diff/starting_volume)
        
        exponent = -inwards_flow/outwards_flow_diff
        
        value = inwards_concentration - inwards_concentration * (correction_base**exponent)
        
        return(value)
    
    
    
    def compute_concentrations_for_conditions(self, time_in_the_series, inwards_flow, outwards_flow, inwards_concentration, starting_concentration, starting_volume):   
            
        if inwards_flow == outwards_flow:     
            inwards_flow = inwards_flow + 0.00001
        
        return (inwards_concentration / (inwards_concentration - starting_concentration) - (1 + (inwards_flow - outwards_flow) * time_in_the_series / starting_volume) ** (inwards_flow / (outwards_flow - inwards_flow))) * (inwards_concentration - starting_concentration)

        
    
    
    
    def compute_concentrations(self):
        start = 0
        inwards_flow = self.system.series_dictionary["inwards_flow_series"][0]
        inwards_concentration = self.system.series_dictionary["inwards_concentration_series"][0]
        outwards_flow = self.system.series_dictionary["outwards_flow_series"][0]
        starting_volume = self.system.series_dictionary["volume_series"][0]
        starting_concentration = 0
        
        for ended in range(self.system.time_steps_amount):
            if self.condition_changes_time_steps[ended]==False:
                continue
            else:
                time_array = self.system.series_dictionary["time_series"][start:ended]
                time_array = time_array - self.system.series_dictionary["time_series"][start]
                concentration_array= self.compute_concentrations_for_conditions(time_array, inwards_flow, outwards_flow, inwards_concentration, starting_concentration, starting_volume)
                self.system.series_dictionary["outwards_concentration_series"][start:ended] = concentration_array
                
                
                start = ended - 1
                inwards_flow = self.system.series_dictionary["inwards_flow_series"][ended-1]
                inwards_concentration = self.system.series_dictionary["inwards_concentration_series"][ended-1]
                outwards_flow = self.system.series_dictionary["outwards_flow_series"][ended-1]
                starting_volume = self.system.series_dictionary["volume_series"][ended-1]
                starting_concentration = self.system.series_dictionary["outwards_concentration_series"][ended-1]
            
            
        time_array = self.system.series_dictionary["time_series"][start:self.system.time_steps_amount]
        time_array = time_array - self.system.series_dictionary["time_series"][start]
        concentration_array= self.compute_concentrations_for_conditions(time_array, inwards_flow, outwards_flow, inwards_concentration, starting_concentration, starting_volume)
        self.system.series_dictionary["outwards_concentration_series"][start:self.system.time_steps_amount] = concentration_array    
            
        
        
    def simulate_physics(self):
        self.compute_volume()
        self.compute_concentrations()
        
        

    
        # Xa = 1-(1+Ta*R)**-R^-1
        # R = (Qe-Qs)/Qe
        # Ta = (Qe/Vi) * t         #Dq
        # Xa = Xs/Xe
        # Va= Vi*(1 + R*Ta)
        
        # La adimensionalizacion en este caso resulta ser el reescalado por constantes, debido a esta propiedad, podemos considerar que no importan los cambios de regimen a la hora de adimensionalizar y redimensionalizar debido a que simplemente se estaria escalando por otra constante
    
        
        
        







