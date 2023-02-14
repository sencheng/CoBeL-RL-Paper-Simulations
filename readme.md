## **CoBeL-RL: Simulations** ##

----------------------------

**Install Requirements**

* First, download and install CoBeL-RL:

    `https://github.com/sencheng/CoBeL-RL`
    
* Install python packages required for plotting:

    `pip3 install -r requirements.txt`
    
* Some simulations require Blender2.79b. Download and install Blender2.79b at:  

     `https://download.blender.org/release/Blender2.79/`
     
* Some simulations require Unity. Builds can be found in the resources directory:  

     `resources/unity/online/`

Note : Only Blender v2.79b is supported. Newer versions of Blender will not work with the system.  

------------------------------

**Simulations Guide**  

Simulations can be found in the 'simulations' directory.
The following sections will guide you through how to run specific simulations.

------------------------------

<details>
<summary>
Monitor Demonstration Simulation
</summary>

*  Navigate to `simulations/monitors/`.

*  Run the script `simulation_monitors_behavior.py` to record behavioral data.

*  Run the script `simulation_monitors_representation.py` to record unit activity.

*  Run the script `plot_behavior.py` to plot behavioral data.

*  Run the script `plot_activity.py` to plot unit activity for a set of trials.

*  A demonstration for visualization during a simulation run is provided in `demo_monitors_behavior.py`.

</details>

------------------------------

<details>
<summary>
Extinction Learning Simulation
</summary>

*  Navigate to `simulations/extinction/`.

*  Run the script `simulation_gridworld.py` for the gridworld simulation.

*  Run the script `plot_gridworld.py` to plot the results of the gridworld simulation.

*  Run the script `simulation_unity.py` for the unity simulation.

*  Run the script `plot_unity.py` to plot the results of the unity simulation.

</details>

------------------------------

<details>
<summary>
Latent Learning Simulation
</summary>

*  Navigate to `simulations/latent_learning/`.

*  Run the script `simulation_latent_learning.py` for the latent learning simulation.

*  Run the script `plot_escape_latency.py` to plot the escape latency results of the latent learning simulation.

*  Run the script `plot_SR.py` to plot the SR results of the latent learning simulation.

*  A demonstration with visualization during a simulation run is provided in `demo_latent_learning.py`.

</details>

------------------------------

<details>
<summary>
Online and Offline Simulation
</summary>

*  Navigate to `simulations/online_offline/`.

*  Set the unity path variable.

*  Run the script `simulation_unity.py` for the unity simulation.

*  Run the script `plot_unity.py` to plot the results of the unity simulation.

</details>

------------------------------

<details>
<summary>
Place Field Simulation
</summary>

*  Navigate to `simulations/place_fields/`.

*  Set the Blender path variable.

*  Run the script `simulation_place_fields.py` for the place field simulation.

*  Run the script `plot_place_fields.py` to plot the results of the place field simulation.

</details>

------------------------------   

<details>
<summary>
Decision Task Simulation
</summary>

*  Navigate to `simulations/choice/`.

*  Run the script `simulation_choice.py` for the choice task simulation.

*  Run the script `plot_choice.py` to plot the results of the choice task simulation.

</details>

------------------------------  
