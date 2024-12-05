#TODO name graphs


from SystemSeries import *
from AnalyticalPhysicsSimulator import *


def generate_original_series():

    system_series_list = []
    a_physics_simulator = AnalyticalPhysicsSimulator()

    
    a_system = SystemSeries()
    a_system = SystemSeries()
    a_system.set_time_conditions(300, 1)
    a_system.set_starting_volume_conditions(30)
    a_system.set_steady_conditions("inwards_concentration_series", 2)
    a_system.set_steady_conditions("inwards_flow_series", 2)
    a_system.set_steady_conditions("outwards_flow_series", 1)
    a_system.simulate_physics(a_physics_simulator)

    system_series_list.append(a_system)
    
    
    
    a_system = SystemSeries()
    a_system = SystemSeries()
    a_system.set_time_conditions(150, 1)
    a_system.set_starting_volume_conditions(50)
    a_system.set_steady_conditions("inwards_concentration_series", 6, ended_time_step=50)
    a_system.set_steady_conditions("inwards_concentration_series", 4, starting_time_step=50)
    a_system.set_steady_conditions("inwards_flow_series", 3)
    a_system.set_steady_conditions("outwards_flow_series", 2)
    a_system.simulate_physics(a_physics_simulator)

    system_series_list.append(a_system)
    
    
    a_system = SystemSeries()
    a_system = SystemSeries()
    a_system.set_time_conditions(260, 1)
    a_system.set_starting_volume_conditions(20)
    a_system.set_steady_conditions("inwards_concentration_series", 4, ended_time_step=100)
    a_system.set_steady_conditions("inwards_concentration_series", 6, starting_time_step=100)
    a_system.set_steady_conditions("inwards_flow_series", 6, ended_time_step=100)
    a_system.set_steady_conditions("inwards_flow_series", 1.5, starting_time_step=100)
    a_system.set_steady_conditions("outwards_flow_series", 0.75)
    a_system.simulate_physics(a_physics_simulator)

    system_series_list.append(a_system)
    
    
    a_system = SystemSeries()
    a_system = SystemSeries()
    a_system.set_time_conditions(180, 1)
    a_system.set_starting_volume_conditions(60)
    a_system.set_steady_conditions("inwards_concentration_series", 1)
    a_system.set_steady_conditions("inwards_flow_series", 6, ended_time_step=120)
    a_system.set_steady_conditions("inwards_flow_series", 2, starting_time_step=120)
    a_system.set_steady_conditions("outwards_flow_series", 4, ended_time_step=60)
    a_system.set_steady_conditions("outwards_flow_series", 2, starting_time_step=60, ended_time_step=120)
    a_system.set_steady_conditions("outwards_flow_series", 0.5, starting_time_step=120)
    a_system.simulate_physics(a_physics_simulator)

    system_series_list.append(a_system)
    
    
    
    a_system = SystemSeries()
    a_system = SystemSeries()
    a_system.set_time_conditions(180, 1)
    a_system.set_starting_volume_conditions(60)
    a_system.set_steady_conditions("inwards_concentration_series", 3)
    a_system.set_steady_conditions("inwards_flow_series", 3)
    a_system.set_steady_conditions("inwards_flow_series", 2, starting_time_step=120)
    a_system.set_steady_conditions("outwards_flow_series", 3)

    a_system.simulate_physics(a_physics_simulator)

    system_series_list.append(a_system)
    

    return(system_series_list)
    


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    
    
    system_series_list=generate_original_series()
    for serie in system_series_list:
        plt.plot(serie.series_dictionary["outwards_concentration_series"])
    