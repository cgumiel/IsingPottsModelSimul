import sys
import subprocess
import random
import math

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
try:
    from pydataset import data
except:
    print("Installing missing pydataset ")
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--user', 'pydataset'])

try:
    import re
except:
    print("Installing missing re ")
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--user', 're'])


# # librerías de visualizaciones
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

try:
    import sklearn as sk
except:
    print("Installing missing sklearn ")
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--user', 'sklearn'])

L=0
t=2.26
nblock=1
nsamp = 1000
ntherm = random.randint(1,32500)
seed = random.randint(1,32500)

def step(spin_input, t_input, J_input, B_input, L_input):
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
    DeltaE = 2 * (J_input*(spin_input * vecinos) + B_input*spin_input)
    #calculate the transition probabilities
    p_trans = np.exp(-DeltaE/t_input)
    # Decide wich transition will occur
    transitions = ((np.random.rand(L_input, L_input) < p_trans) & np.random.randint(0, 2, [L_input, L_input]))*-2 +1
    #print (transitions) 
    spin = np.multiply(spin_input, transitions)

    return (spin, DeltaE)

def clicked():
    
    L = int(txt_length.get())
    t = float(txt_temperature.get())
    ntherm = int(txt_steps_thermal.get())
    J= float(txt_J.get())
    B=0

    print ("2D Ising model with the Metropolis algorithm.\n")
    print("\n====    ", L, " x ", L, "     T = ", t,"    ====\n")
    print("\nntherm  nblock   nsamp   seed\n")
    print(ntherm, nblock, nsamp, seed)
   
    print("\n energy      cv        magn     \n")

    # Animation parameters
    ax = plt.subplot()

    spin_in = np.ones((L,L))
    print(spin_in)
    # Thermalize the system 
    for i in range (0,ntherm):
        (spin_in, DeltaE) = step(spin_in, t, J, B, L)

        ax.cla()
        ax.imshow(spin_in)
        plt.pause(0.0001)    

def clicked_loop():
    
    L = int(txt_length.get())
    t = float(txt_temperature.get())
    ntherm = int(txt_steps_thermal.get())
    t_max = float(txt_temperature_max.get())
    t_steps = float(txt_steps_t.get())
    L2 = L*L
    J= float(txt_J.get())
    B=0

    print ("2D Ising model with the Metropolis algorithm.\n")
    print("\n====    ", L, " x ", L, "     T = ", t,"    ====\n")
    print("\nntherm  nblock   nsamp   seed\n")
    print(ntherm, nblock, nsamp, seed)
   
    print("\n energy      cv        magn     \n")

    # Animation paramet, ers
    fig, ax = plt.subplots()
    spin_in = np.random.randint(0, 2, [L, L]) * 2 - 1
    print(spin_in)
    # Thermalize the system 
    M = []
    M_square_mean = []
    E = []
    E_square_mean = []  #E^2
    C = []  #Specific Heat
    ts = []
    kappa = []
    m_theoretical = []
    c_theoretical = []
    kappa_theoretical = []
    minimum_t_step = int((t/t_steps))
    maximum_t_step = int((t_max/t_steps))
    t_critical_theo = 2.26919
    c_plus = 0.96258
    c_minus = 0.02554
    for t_index in range (minimum_t_step,maximum_t_step):
        current_t = t_index*t_steps
        if current_t < t_critical_theo:
            current_m_theo = math.pow(1-math.pow(math.sinh(2/current_t),-4),(1/8))
            #current_kappa_theo = 1/t_critical_theo*c_minus*math.pow((t_critical_theo-current_t)/t_critical_theo,-1*(7/4))
            #kappa_theoretical.append(current_kappa_theo)
        else:
            current_m_theo = 0
            #current_kappa_theo = 1/t_critical_theo*c_plus*math.pow((current_t - t_critical_theo)/t_critical_theo,-1*(7/4))
            #kappa_theoretical.append(current_kappa_theo)
        current_c_theo = -0.4945*math.log(abs((t_critical_theo-current_t)/t_critical_theo))

        m_theoretical.append(current_m_theo)
        c_theoretical.append(current_c_theo)
        
        M_acum = 0
        M_sq_acum = 0
        E_acum = 0
        E_sq_acum =0
        ts.append(current_t)
        spin_in = np.ones((L,L))
        #execute step`s
        for i in range (0,ntherm):
            (spin_in, DeltaE) = step(spin_in, current_t, J, B, L)
        print("Temperature: ",current_t,"execution: ",i)
        for i in range (0,ntherm):
            (spin_in, DeltaE) = step(spin_in, current_t, J, B, L)

            current_M = sum(sum(spin_in))/L2  
            M_acum = M_acum + abs(current_M)
            spin_sq = np.multiply(spin_in, spin_in)
            current_M_sq = sum(sum(spin_sq))/L2 
            M_sq_acum = M_sq_acum + current_M_sq

            current_E = -sum(sum(DeltaE))/(L2) # Divide by two because of double counting
            E_acum = E_acum + current_E
            DeltaE_sq = np.multiply(DeltaE, DeltaE)
            current_E_sq = sum(sum((DeltaE_sq)))/(L2)
            E_sq_acum = E_sq_acum + current_E_sq

        #compute thermodinamical variables
        #magnetizaton
        M.append(current_M)
        #Kappa, magnetic subceptibility
        current_M_mean = M_acum/ntherm
        current_M_sq_mean = M_sq_acum/ntherm
        M_square_mean.append(current_M_sq_mean)
        kappa.append(1/(current_t)*(current_M_sq_mean-(current_M_mean*current_M_mean)))
        
        #E= -1/2DeltaE
        E.append(current_E) 
        #specific heat
        current_E_mean = E_acum/ntherm
        current_E_sq_mean = E_sq_acum/ntherm
        E_square_mean.append(current_E_sq_mean) 
        C.append(1/((current_t*current_t))*(current_E_sq_mean-(current_E_mean*current_E_mean)))
 
    df = pd.DataFrame()
    df.insert(0,'ts',ts)
    df.insert(1,'m',M)
    df.insert(2,'m_theoretical',m_theoretical)
    df.insert(3,'E',E)
    df.insert(4,'C',C)
    df.insert(5,'c_theoretical',c_theoretical)
    df.insert(6,'kappa',kappa)
    df.insert(7,'E_square',E_square_mean)
    df.insert(8,'M_square',M_square_mean)
    df.to_csv('ising.csv')

    # ejecuta el clasificador
    #clf = sk.pot .LinearRegression()
    #clf.fit(X, Y)

    plt.subplot(2,2,1)
    plt.title('m=f(t)') 
    plt.plot(ts,M)
    plt.plot(ts,m_theoretical)

    plt.subplot(2,2,2) 
    plt.title('e = f(t)') 
    plt.plot(ts,E)

    plt.subplot(2,2,3) 
    plt.title('C_v = f(t)') 
    plt.plot(ts,C)    
    plt.plot(ts,c_theoretical)

    plt.subplot(2,2,4) 
    plt.title('Magnetization = f(t)') 
    plt.plot(ts,kappa)  
    #plt.plot(ts,kappa_theoretical)

    plt.show()
    
    

def clicked_renorm():
    
    L = int(txt_length.get())
    t = float(txt_temperature.get())
    ntherm = int(txt_steps_thermal.get())
    t_max = float(txt_temperature_max.get())
    t_steps = float(txt_steps_t.get())
    L2 = L*L
    J= float(txt_J.get())
    B=0
    #size of the renormalizaton cell
    l_celda = int(txt_size_renorm.get())

    print ("2D Ising model with the Metropolis algorithm.\n")
    print("\n====    ", L, " x ", L, "     T = ", t,"    ====\n")
    print("\nntherm  nblock   nsamp   seed\n")
    print(ntherm, nblock, nsamp, seed)
   
    print("\n energy      cv        magn     \n")

    # Animation paramet, ers
    fig, ax = plt.subplots()
    spin_in = []
    #spin =  [[1] * L for i in range(L)]
    spin_in = np.random.randint(0, 2, [L, L]) * 2 - 1
    # Thermalize the system 
    M = []
    E = []
    ts = []
    m_theoretical = []
    minimum_t_step = int((t/t_steps))
    maximum_t_step = int((t_max/t_steps))
    for t_index in range (minimum_t_step,maximum_t_step):
        current_t = t_index*t_steps
        if current_t < 2.269:
            current_m = math.pow(1-math.pow(math.sinh(2/current_t),-4),(1/8))
        else:
            current_m = 0
        m_theoretical.append(current_m)
        ts.append(current_t)
        #spin_in = np.random.randint(0, 2, [L, L]) * 2 - 1
        spin_in = np.ones((L,L))
        for i in range (0,ntherm):
            (spin_in, DeltaE) = step(spin_in, current_t, J, B, L)

        print("Temperature: ",current_t,"execution: ",i)        
        M.append(sum(sum(spin_in))/L2)
        E.append(-sum(sum(DeltaE))/(2*L2)) # Divide by two because of double counting
 
    plt.subplot(1,2,1)
    plt.title('m=f(t)') 
    plt.plot(ts,M)
    #print (m_theoretical)
    plt.plot(ts,m_theoretical)
    plt.subplot(1,2,2) 
    plt.title('e = f(t)') 
    plt.plot(ts,E)
    plt.show()       
    

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

cb_renormalization = tk.IntVar()
C1 = tk.Checkbutton(window, text = "Renormalization", variable = cb_renormalization, \
                 onvalue = 1, offvalue = 0)
C1.grid(column=0, row =4)

lbl_size_renorm = tk.Label(window, text="Renormalization length")
lbl_size_renorm.grid(column=0, row=5) 
txt_size_renorm = tk.Entry(window,width=10)
txt_size_renorm.grid(column=1, row=5)   
txt_size_renorm.insert(0, 2)  

btn = tk.Button(window, text="Start Simulation only minimum temperature", command=clicked)
btn.grid(column=0, row=6)

btn_loop = tk.Button(window, text="Start continuous simulation", command=clicked_loop)
btn_loop.grid(column=1, row=6)

btn_renorm = tk.Button(window, text="Continuous simulation with renorm", command=clicked_renorm)
btn_renorm.grid(column=2, row=6)

window.mainloop()
