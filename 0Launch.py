import sys
import subprocess
import random
import math

from datetime import datetime
try:
    import numpy as np
except:
    print("Installing missing numpy ")
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--user', 'numpy'])

try:
   import pandas as pd
except:
    print("Installing missing pandas ")
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--user', 'pandas'])

# Visualization libraries
try:
    import matplotlib.pyplot as plt 
except:
    print("Installing missing matplotlib ")
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--user', 'matplotlib'])

try:
    import  tkinter as tk
except:
    print("Installing missing tkinter ")
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--user', 'tkinter'])

L=0
t=2.26
nblock=1
nsamp = 1000
ntherm = random.randint(1,32500)
seed = random.randint(1,32500)

############################ INITIALISE VARIABLES ############################

def initialise():
    L = int(txt_length.get())
    t = float(txt_temperature.get())
    ntherm = int(txt_steps_thermal.get())
    J= float(txt_J.get())
    B=0
    t_max = float(txt_temperature_max.get())
    t_steps = float(txt_steps_t.get())
    L2 = L*L

    return (L, t, t_max, t_steps, L2, ntherm, J, B)
    
############################ ISING MODEL ############################

def compute_energy_ising(spin, J, B, L):
    # Calculating the total spin of neighbouring cells
    spin_anterior_x = np.roll(spin,-1, axis=1)
    spin_anterior_y = np.roll(spin,-1, axis=0)
    spin_posterior_x = np.roll(spin,1, axis=1)
    spin_posterior_y = np.roll(spin,1, axis=0)
    vecinos = spin_anterior_x + spin_anterior_y + spin_posterior_x + spin_posterior_y

    E_spin_J = -J*(np.multiply(spin, vecinos))
    E_spin_B = -B*spin

    E_spin = E_spin_J + E_spin_B
    
    L2 = L*L
    E = sum(sum(E_spin))/(2*L2) #divide by 2 because of double counting

    return (E, E_spin)
    
def step_metro_ising(spin_input, t_input, J_input, B_input, L_input):
    # Calculating the total spin of neighbouring cells
    spin_anterior_x = np.roll(spin_input,-1, axis=1)
    spin_anterior_y = np.roll(spin_input,-1, axis=0)
    spin_posterior_x = np.roll(spin_input,1, axis=1)
    spin_posterior_y = np.roll(spin_input,1, axis=0)

    #E = -J*SUM(s(i,j)*(s(i-1,j)+s(i+1,j)+s(i, j-1)+s(i,j+1))) -H SUM(s(i,j))
    # definiendo vecinos(i,j) = (s(i-1,j)+s(i+1,j)+s(i, j-1)+s(i,j+1))
    #E = -J*SUM(s(i,j)*vecinos(i,j)) -H SUM(s(i,j))
    vecinos = spin_anterior_x + spin_anterior_y + spin_posterior_x + spin_posterior_y
    #Calculate the change in energy of flipping a spin
    E_spin_J = -J_input*(np.multiply(spin_input, vecinos))
    E_spin_B = B_input*spin_input
    DeltaE = -2 * (E_spin_J) + E_spin_B
    #calculate the transition probabilities
    p_trans = np.exp(-DeltaE/t_input)
    # Decide wich transition will occur
    transitions = ((np.random.rand(L_input, L_input) < p_trans) & np.random.randint(0, 2, [L_input, L_input]))*-2 +1
    #print (transitions) 
    spin = np.multiply(spin_input, transitions)
    
    return (spin)

def generate_trying_spin_wolff_ising(input_spin):
    trying_spin = input_spin*-1
    return(trying_spin)    

def compute_theoretical_values_ising(J_input, minimum_t_step, maximum_t_step, t_steps):

    m_theoretical = []
    c_theoretical = []
    kappa_theoretical = []
    t_critical_theo = 2.26919
    c_plus = 0.96258
    c_minus = 0.02554

    for t_index in range (minimum_t_step,maximum_t_step):
        current_t = t_index*t_steps
        if current_t < t_critical_theo:
            current_m_theo = math.pow(1-math.pow(math.sinh(2/current_t),-4),(1/8))
            current_kappa_theo = 1/t_critical_theo*c_minus*math.pow((t_critical_theo-current_t)/t_critical_theo,-1*(7/4))
        else:
            current_m_theo = 0
            current_kappa_theo = 1/t_critical_theo*c_plus*math.pow((current_t - t_critical_theo)/t_critical_theo,-1*(7/4))

        current_c_theo = -0.4945*math.log(abs((t_critical_theo-current_t)/t_critical_theo))

        m_theoretical.append(current_m_theo)
        c_theoretical.append(current_c_theo)
        kappa_theoretical.append(current_kappa_theo)
        
        
    return (m_theoretical, c_theoretical, kappa_theoretical)

############################ POTTS MODEL ############################

def compute_energy_potts(spin, J, B, L):
    spin_anterior_x = np.roll(spin,-1, axis=1)
    spin_anterior_y = np.roll(spin,-1, axis=0)
    spin_posterior_x = np.roll(spin,1, axis=1)
    spin_posterior_y = np.roll(spin,1, axis=0)

    #Calculate the change in energy of flipping a spin
    E_spin_J = -J*((spin == spin_anterior_x).astype(int) +(spin == spin_anterior_y).astype(int) + \
        (spin == spin_posterior_x).astype(int) + (spin == spin_posterior_y).astype(int))   
    E_spin_B = -B*spin

    E_spin = E_spin_J + E_spin_B
    
    L2 = L*L
    E = sum(sum(E_spin))/(2*L2) #divide by 2 because of double counting

    return (E, E_spin)

def step_metro_potts(spin_input, t_input, J_input, B_input, L_input, q):
    #try a spin with one less value of q
    tryin_spin_first = np.random.randint(0, q-1, [L_input, L_input])
    trying_spin = np.mod(spin_input + tryin_spin_first+1,q)
    # Calculating the total spin of neighbouring cells
    spin_anterior_x = np.roll(spin_input,-1, axis=1)
    spin_anterior_y = np.roll(spin_input,-1, axis=0)
    spin_posterior_x = np.roll(spin_input,1, axis=1)
    spin_posterior_y = np.roll(spin_input,1, axis=0)

    #Calculate the change in energy of flipping a spin
    E_spin_J2 = -J_input*((trying_spin == spin_anterior_x).astype(int) +(trying_spin == spin_anterior_y).astype(int) + \
        (trying_spin == spin_posterior_x).astype(int) + (trying_spin == spin_posterior_y).astype(int))
    E_spin_J1 = -J_input*((spin_input == spin_anterior_x).astype(int) +(spin_input == spin_anterior_y).astype(int) + \
        (spin_input == spin_posterior_x).astype(int) + (spin_input == spin_posterior_y).astype(int))
    
    DeltaE = E_spin_J2 -E_spin_J1 
    #calculate the transition probabilities
    p_trans = np.exp(-DeltaE/t_input)
    # Decide wich transition will occur
    transitions = ((np.random.rand(L_input, L_input) < p_trans) & np.random.randint(0, 2, [L_input, L_input]))
    no_transitions = (transitions*-1)+1
    #print (transitions) 
    spin1 = np.multiply(trying_spin, transitions)
    spin2 = np.multiply(spin_input, no_transitions)

    spin = spin1+spin2

    return (spin)

def generate_trying_spin_wolff_potts(input_spin, q):
    new_spin = random.randint(0, q-2)
    trying_spin = np.mod(input_spin + 
    +1,q)
    return(trying_spin)

def compute_theoretical_values_potts(J_input, q, minimum_t_step, maximum_t_step, t_steps):

    m_theoretical = []
    c_theoretical = []
    kappa_theoretical = []
    t_critical_theo = 2.26919
    c_plus = 0.96258
    c_minus = 0.02554
    t_critical_theo_potts = J_input*(math.log(1+math.sqrt(q)))**(-1)
    print (t_critical_theo_potts)    

    for t_index in range (minimum_t_step,maximum_t_step):
        current_t = t_index*t_steps
        if current_t < t_critical_theo:
            current_m_theo = math.pow(1-math.pow(math.sinh(2/current_t),-4),(1/8))
            current_kappa_theo = 1/t_critical_theo*c_minus*math.pow((t_critical_theo-current_t)/t_critical_theo,-1*(7/4))
        else:
            current_m_theo = 0
            current_kappa_theo = 1/t_critical_theo*c_plus*math.pow((current_t - t_critical_theo)/t_critical_theo,-1*(7/4))

        current_c_theo = -0.4945*math.log(abs((t_critical_theo-current_t)/t_critical_theo))



        m_theoretical.append(current_m_theo)
        c_theoretical.append(current_c_theo)
        kappa_theoretical.append(current_kappa_theo)
        
        
    return (m_theoretical, c_theoretical, kappa_theoretical)

############################ PERFORM A STEP USING WOLFF aLGORITHM ############################
############################        Both Ising and Potts          ############################

def step_wolff(spin_input, t_input, J_input, B_input, L_input, is_potts, q=0):

    # 1. Choose a single site of the lattice for starting to build the cluster, in a random way.
    x = np.random.randint(0, L_input)
    y = np.random.randint(0, L_input)

    sign = spin_input[x, y]
    if (is_potts):
        trying_spin = generate_trying_spin_wolff_potts(sign, q)
    else:  #Ising
        trying_spin = generate_trying_spin_wolff_ising(sign)

    P_add = 1 - math.exp(-2 * J_input / t_input)
    stack = [[x, y]]
    lable = np.ones([L_input, L_input], int)
    lable[x, y] = 0

    # 2. Consider all the links connected to that initial site; the $\ell_{-}$ links are never to be activated; 
    #    activate the $\ell_{+}$ links with the probability $p_{+}=1-\exp(-2\beta\psi_{-}\psi_{+})$, thus forming a first cluster of sites, 
    #    that is, updating the cluster from a single site to possibly a few sites.
    while len(stack) > 0.5:

        # While stack is not empty, pop and flip a spin

        [currentx, currenty] = stack.pop()
        spin_input[currentx, currenty] = trying_spin

        # Append neighbor spins

        # Left neighbor

        leftx = (currentx-1)%L_input
        lefty = currenty

        if (spin_input[leftx, lefty] == sign) and (lable[leftx, lefty]) and (np.random.rand() < P_add):
            stack.append([leftx, lefty])
            lable[leftx, lefty] = 0

        # Right neighbor

        rightx = (currentx+1)%L_input
        righty = currenty

        if (spin_input[rightx, righty] == sign) and (lable[rightx, righty]) and (np.random.rand() < P_add):
            stack.append([rightx, righty])
            lable[rightx, righty] = 0

        # Up neighbor

        upx = currentx
        upy = (currenty-1)%L_input

        if (spin_input[upx, upy] == sign) and (lable[upx, upy]) and (np.random.rand() < P_add):
            stack.append([upx, upy])
            lable[upx, upy] = 0

        # Down neighbor

        downx = currentx
        downy = (currenty+1)%L_input

        if (spin_input[downx, downy] == sign) and lable[downx, downy] and np.random.rand() < P_add:
            stack.append([downx, downy])
            lable[downx, downy] = 0

    # cluster size

    cluster_size = L_input * L_input - sum(sum(lable))

    return (spin_input, cluster_size)       

############################ PLOT OBSERVABLES ############################

def plot_and_save(ts, M, E, C, kappa, is_metropolis, is_potts, J, L, ntherm, minimum_t_step, maximum_t_step, t_steps, q = 0):
    is_ising = not is_potts
    if is_potts:
        (m_theoretical, c_theoretical, kappa_theoretical) = compute_theoretical_values_potts(J, q, minimum_t_step, maximum_t_step, t_steps)
    else:
        (m_theoretical, c_theoretical, kappa_theoretical) = compute_theoretical_values_ising(J, minimum_t_step, maximum_t_step, t_steps)

    plt.tight_layout()
    plt.subplot(2,2,1)
    plt.plot(ts,M)
    
    #plt.title('m=f(t)')
    if is_ising: 
        plt.plot(ts, m_theoretical)
        plt.title("J = {} L = {} ".format(J, L))
    else:
        plt.title("q = {} J = {} L = {} ".format(q, J, L))
    plt.xlabel('Temperature') 
    plt.ylabel('Magnetization per spin') 

    #plt.plot(ts,m_theoretical)

    plt.subplot(2,2,2) 
    plt.plot(ts,E)
    #plt.title('e = f(t)') 
    plt.title("steps = {} montecarlo = {}".format(ntherm, is_metropolis))
    plt.xlabel('Temperature') 
    plt.ylabel('Energy per spin') 


    plt.subplot(2,2,3) 
    plt.plot(ts,C)    
    if is_ising: 
        plt.plot(ts, c_theoretical)
    #plt.title('Specific heat = f(t)')
    plt.xlabel('Temperature') 
    plt.ylabel('Specific Heat per spin')  
    
    #plt.plot(ts,c_theoretical)

    plt.subplot(2,2,4) 
    plt.plot(ts,kappa)  
    if is_ising: 
        plt.plot(ts,kappa_theoretical)
    #plt.title('Susceptibility = f(t)') 
    plt.xlabel('Temperature') 
    plt.ylabel('Magnetic Susceptibility per spin') 
    if is_ising:
        title = ("Ising_J{}_L{}steps{}montecarlo_{}_{}.jpg".format(J, L,ntherm, is_metropolis, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    else:
        title = ("Potts_q{}_J{}_L{}steps{}montecarlo_{}_{}.jpg".format(q, J, L,ntherm, is_metropolis, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    plt.savefig(title, bbox_inches='tight')
    #plt.plot(ts,kappa_theoretical)

    plt.show()

    df = pd.DataFrame()
    df.insert(0,'ts',ts)
    df.insert(1,'m',M)
    df.insert(2,'m_theoretical',m_theoretical)
    df.insert(3,'E',E)
    df.insert(4,'C',C)
    df.insert(5,'c_theoretical',c_theoretical)
    df.insert(6,'kappa',kappa)
    df.insert(7,'kappa_theoretical',kappa_theoretical)
    df.insert(8,'E_square',E_square_mean)
    df.insert(9,'M_square',M_square_mean)        

    df.to_csv("{}_{}_J{}_L{}steps{}_{}.csv".format(model, algorithm, J, L,ntherm, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

############################ PERFORM SIMULATION AND VISUALISE SPINS AT FIXED T ############################
def simul_system_fixed_temperature(is_metropolis, is_potts):
    
    (L, t, t_max, t_steps, L2, ntherm, J, B) = initialise()
    if (is_potts):
        q=int(txt_size_potts.get())
        model = "Potts q={}".format(q)
        spin_in = np.random.randint(0, q, [L, L])
    else:
        model = "Ising"
        spin_in = np.random.randint(0, 2, [L, L]) * 2 - 1
        q=0
    if is_metropolis:
        algorithm = "metropolis"
 
    else:
        algorithm = "Wolff"
        

    print ("2D {} model with the {} algorithm.\n".format(model, algorithm))
    print("\n====    ", L, " x ", L, "     T = ", t,"    ====\n")
    print("\nntherm  nblock   nsamp   seed\n")
    print(ntherm, nblock, nsamp, seed)
   
    # Animation parameters
    ax = plt.subplot()
    print(spin_in)
    # Execute simulation
    for i in range (0,ntherm):
        if is_metropolis:
            if is_potts:
               (spin_in) = step_metro_potts(spin_in, t, J, B, L, q)
            else: #Ising
               (spin_in) = step_metro_ising(spin_in, t, J, B, L)
        else: #Wolff
            (spin_in, cluster_size) = step_wolff(spin_in, t, J, B, L, is_potts, q)

        ax.cla()
        ax.set_title("{} model with {} algorithm\n T = {}, J = {} \nFrame {}".format(model, algorithm, t, J,i))
        ax.imshow(spin_in)
        plt.pause(0.00001)

############################ PERFORM SIMULATION IN A RANGE OF TS AND COMPUTE THERMODINAMIC VARIABLES ############################
def simul_system_temperature_range(is_metropolis, is_potts):

    (L, t, t_max, t_steps, L2, ntherm, J, B) = initialise()

    if (is_potts):
        q=int(txt_size_potts.get())
        model = "Potts q={}".format(q)
        spin_in = np.random.randint(0, q, [L, L])
    else: #Ising
        model = "Ising"
        spin_in = np.random.randint(0, 2, [L, L]) * 2 - 1
        q=0
    if is_metropolis:
        algorithm = "metropolis"
    else: #Wolff
        algorithm = "Wolff"

    print ("2D {} model with the {} algorithm.\n Temperature Range".format(model, algorithm))
    print("\n====    ", L, " x ", L, "     T = ", t,"    ====\n")
    print("\nntherm  nblock   nsamp   seed\n")
    print(ntherm, nblock, nsamp, seed)    

    print(spin_in)
    # Thermalize the system 
    M = []
    M_square_mean = []
    E = []
    E_square_mean = []  #E^2
    C = []  #Specific Heat
    ts = []
    kappa = []
    minimum_t_step = int((t/t_steps))
    maximum_t_step = int((t_max/t_steps))
    
    for t_index in range (minimum_t_step,maximum_t_step):
        current_t = t_index*t_steps
        
        M_acum = 0
        M_sq_acum = 0
        E_acum = 0
        E_sq_acum =0
        ts.append(current_t)
        spin_in = np.ones((L,L))*-1

        #Thermalise the system
        for i in range (0,ntherm):
            if is_metropolis:
                if is_potts:
                    (spin_in) = step_metro_potts(spin_in, current_t, J, B, L, q)
                else: #Ising
                    (spin_in) = step_metro_ising(spin_in, current_t, J, B, L)
            else: #Wolff
                (spin_in, cluster_size) = step_wolff(spin_in, current_t, J, B, L, is_potts, q)

        #Execute simulation 
        for i in range (0,ntherm):
            if is_metropolis:
                if is_potts:
                    (spin_in) = step_metro_potts(spin_in, current_t, J, B, L, q)
                else: #Ising
                    (spin_in) = step_metro_ising(spin_in, current_t, J, B, L)
            else: #Wolff
                (spin_in, cluster_size) = step_wolff(spin_in, current_t, J, B, L, is_potts, q)    

            if is_potts:
                Ni = []
                for index in range (0,q):
                    Ni.append((spin_in == index).sum())
                current_M = q* (max(Ni)/L2-1)/(q-1)     
                
                (current_E, E_spin) = compute_energy_potts(spin_in, current_t, B, L)
 
            else:  #Ising
                current_M = abs(sum(sum(spin_in)))/L2        
                (current_E, E_spin) = compute_energy_ising(spin_in, J, B, L)  

            M_acum = M_acum + abs(current_M)
            spin_sq = np.multiply(spin_in, spin_in)
            #current_M_sq = sum(sum(spin_sq))/L2 
            current_M_sq = current_M*current_M
            M_sq_acum = M_sq_acum + current_M_sq
                        
            E_acum = E_acum + current_E
            DeltaE_sq = np.multiply(E_spin, E_spin)
            #current_E_sq = sum(sum((DeltaE_sq)))/(4*L2)
            current_E_sq = current_E*current_E
            E_sq_acum = E_sq_acum + current_E_sq   
        print("Temperature: ",current_t,"execution: ",i)
        #compute thermodinamical variables
        #magnetizaton
        M.append(current_M)
        #Kappa, magnetic subceptibility
        current_M_mean = M_acum/(ntherm+1)
        current_M_sq_mean = M_sq_acum/(ntherm+1)

        M_square_mean.append(current_M_sq_mean)
        kappa.append(1/(current_t)*(current_M_sq_mean-(current_M_mean**2)))
        
        #E= -1/2DeltaE
        E.append(current_E) 
        #specific heat
        current_E_mean = E_acum/(ntherm+1)
        current_E_sq_mean = E_sq_acum/(ntherm+1)
        E_square_mean.append(current_E_sq_mean) 
        C.append(1/((current_t*current_t))*(current_E_sq_mean-(current_E_mean**2)))

    plot_and_save(ts, M, E, C, kappa, is_metropolis, is_potts, J, L, ntherm, minimum_t_step, maximum_t_step, t_steps, q)

############################ BUTTOMS CALLED FUNCTIONS ############################
def clicked():
    simul_system_fixed_temperature(cb_montecarlo_condition.get(), False)

def clicked_loop():
    simul_system_temperature_range(cb_montecarlo_condition.get(), False)
 
def clicked_potts():
     simul_system_fixed_temperature(cb_montecarlo_condition.get(), True)

def clicked_loop_potts():
    simul_system_temperature_range(cb_montecarlo_condition.get(), True)

print("Starting simulation")

window = tk.Tk()
window.title("Ising Model")
window.geometry('800x200')

lbl_length = tk.Label(window, text="Grid length (L)")
lbl_length.grid(column=0, row=0) 
txt_length = tk.Entry(window,width=10)
txt_length.grid(column=0, row=1) 
txt_length.insert(0,100)

lbl_steps_thermal = tk.Label(window, text="Steps")
lbl_steps_thermal.grid(column=1, row=0) 
txt_steps_thermal = tk.Entry(window,width=10)
txt_steps_thermal.grid(column=1, row=1)    
txt_steps_thermal.insert(0, 100)

lbl_J = tk.Label(window, text="J")
lbl_J.grid(column=2, row=0) 
txt_J = tk.Entry(window,width=10)
txt_J.grid(column=2, row=1)    
txt_J.insert(0, 1)

lbl_temperature = tk.Label(window, text="Normalized Temperature [Minimum]")
lbl_temperature.grid(column=0, row=2) 
txt_temperature = tk.Entry(window,width=10)
txt_temperature.grid(column=0, row=3)    
txt_temperature.insert(0, 1.4)    

lbl_temperature_max = tk.Label(window, text="Normalized Maximum Temperature")
lbl_temperature_max.grid(column=1, row=2) 
txt_temperature_max = tk.Entry(window,width=10)
txt_temperature_max.grid(column=1, row=3)    
txt_temperature_max.insert(0, 3)

lbl_steps_t = tk.Label(window, text="Temperature step increment")
lbl_steps_t.grid(column=2, row=2) 
txt_steps_t = tk.Entry(window,width=10)
txt_steps_t.grid(column=2, row=3)    
txt_steps_t.insert(0, 0.001)    

lbl_temperature_max = tk.Label(window, text="Normalized Maximum Temperature")
lbl_temperature_max.grid(column=1, row=2) 
txt_temperature_max = tk.Entry(window,width=10)
txt_temperature_max.grid(column=1, row=3)    
txt_temperature_max.insert(0, 3)

lbl_steps_t = tk.Label(window, text="Temperature step increment")
lbl_steps_t.grid(column=2, row=2) 
txt_steps_t = tk.Entry(window,width=10)
txt_steps_t.grid(column=2, row=3)    
txt_steps_t.insert(0, 0.01)    

cb_montecarlo_condition = tk.IntVar()
C1 = tk.Checkbutton(window, text = "Select Montecarlo method, Wolff by default", variable = cb_montecarlo_condition, \
                 onvalue = 1, offvalue = 0)
C1.grid(column=0, row =4)

lbl_size_potts = tk.Label(window, text="q potts model")
lbl_size_potts.grid(column=1, row=4) 
txt_size_potts = tk.Entry(window,width=10)
txt_size_potts.grid(column=1, row=5)   
txt_size_potts.insert(0, 2)  

btn = tk.Button(window, text="Ising Simulation only minimum temperature", command=clicked)
btn.grid(column=0, row=7)

btn_loop = tk.Button(window, text="Start continuous simulation", command=clicked_loop)
btn_loop.grid(column=1, row=7)

btn_potts = tk.Button(window, text="Potts simulation only minimum temperature", command=clicked_potts)
btn_potts.grid(column=0, row=8)

btn_potts_loop = tk.Button(window, text="Start Potss continuous simulation", command=clicked_loop_potts)
btn_potts_loop.grid(column=1, row=8)

window.mainloop()
