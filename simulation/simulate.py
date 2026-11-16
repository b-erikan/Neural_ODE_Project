import time
import mujoco
import mujoco.viewer
import numpy as np

from mujoco_simulator import MujocoSimulator

#You can run this script to visualize the MuJoCo simulation in a window. It will launch a passive viewer that displays the simulation in real-time. The simulation will run until you close the viewer window..

def main():

    XML_PATH = "simulation/cart_pole.xml" #Hardcoded path to the XML file for the MuJoCo simulation.
    MAX_PENDULUM_INITIAL_VELOCITY = 10000.0  #Do you wanna see a helicopter? 
    MAX_CART_INITIAL_VELOCITY = 1.0  
    MAX_CART_INITIAL_POSITION = 1.0  


    simulator = MujocoSimulator(XML_PATH)

    simulator.reset() #Maybe this is not necessary but it feels like a good idea to reset the simulation here.


    #Random states
    qpos_initial = [
        np.random.uniform(-MAX_CART_INITIAL_POSITION, MAX_CART_INITIAL_POSITION),
        np.random.uniform(0, 2 * np.pi)
    ]

    qvel_initial = [
        np.random.uniform(-MAX_CART_INITIAL_VELOCITY, MAX_CART_INITIAL_VELOCITY),
        np.random.uniform(-MAX_PENDULUM_INITIAL_VELOCITY, MAX_PENDULUM_INITIAL_VELOCITY)
    ]

    simulator.set_state(qpos_initial, qvel_initial)  # Set the initial state of the simulation
    
    with mujoco.viewer.launch_passive(simulator.model,simulator.data) as viewer:


        while viewer.is_running():
            step_start = time.perf_counter()
            simulator.step()
            viewer.sync()
            #Wait until the next step to maintain real-time simulation speed.
            time_until_next_step = (simulator.model.opt.timestep - (time.perf_counter() - step_start))
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()