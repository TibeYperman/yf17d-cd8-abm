
#Bachelor theis: file containing ABMs
#Author: Tibe Yperman
#E-mail: tibe.yperman@student.uantwerpen.be
#Last revision: 09/05/2026

import numpy as np
import pandas as pd
import random
import multiprocessing as mp

#Visualization
import plotly.graph_objects as go
import plotly.express as px

STATES=["N","MPEC","SLEC","S","C","E","R"]

class CD8Cell():
    """CD8 T cell object with a state attribute"""
    def __init__(self,state="N"):
        self.state=state
        """
        Docstring for __init__
        
        :param state: Type of T cell (N,"MPEC","SLEC",S,C,E,R)
        """


class Model_ABM():
    """Baseline model"""
    def __init__(self,
                 N_begin=1000,
                 A_tpeak=14,A_tsigma=5,A_peak=0.15,
                 slec_fraction=0.4,
                 feedback_c=0.005,
                 b_MPEC=2,b_SLEC=5,
                 d_N=0.0003, d_MPEC=0.02, d_SLEC=0.05, d_S=0.0002, d_C=0.004, d_E=0.01, d_R=0.02,
                 f_S=0.03,f_C=0.05,f_E=0.06,f_R=0.015
                 ):
        """
        Docstring for __init__
        :param N_begin: Initial number of naive T cells
        :param A_tpeak: Time of peak antigen exposure
        :param A_tsigma: Standard deviation of antigen exposure
        :param A_peak: Maximum antigen exposure
        :param slec_fraction: Fraction of cells that follow the SLEC pathway
        :param feedback_c: Feedback constant
        :param b_MPEC: Clonal expansion factor for MPEC cells
        :param b_SLEC: Clonal expansion factor for SLEC cells
        :param d_N: Death rate for naive T cells
        :param d_MPEC: Death rate for MPEC cells
        :param d_SLEC: Death rate for SLEC cells
        :param d_S: Death rate for Tscm cells
        :param d_C: Death rate for Tcm cells
        :param d_E: Death rate for Tem cells
        :param d_R: Death rate for Temra cells
        :param f_S: Differentiation rate from MPEC to Tscm
        :param f_C: Differentiation rate from MPEC to Tcm
        :param f_E: Differentiation rate from SLEC to Tem
        :param f_R: Differentiation rate from SLEC to Temra
        """
        
        #Settings
        self.N_begin=N_begin

        self.A_tpeak=A_tpeak
        self.A_tsigma=A_tsigma
        self.A_peak=A_peak

        self.slec_fraction=slec_fraction

        self.feedback_c=feedback_c

        self.b_MPEC=b_MPEC
        self.b_SLEC=b_SLEC

        self.d_N=d_N
        self.d_MPEC=d_MPEC
        self.d_SLEC=d_SLEC
        self.d_S=d_S
        self.d_C=d_C
        self.d_E=d_E
        self.d_R=d_R

        self.f_S=f_S
        self.f_C=f_C
        self.f_E=f_E
        self.f_R=f_R

        #Data
        self.cells=[]
        self.data=[]

    #Convert values for b_MPEC and b_SLEC to integers
    @property
    def b_MPEC(self):
        return self._b_MPEC
    @b_MPEC.setter
    def b_MPEC(self,value):
        self._b_MPEC=round(value)
    @property
    def b_SLEC(self):
        return self._b_SLEC
    @b_SLEC.setter
    def b_SLEC(self,value):
        self._b_SLEC=round(value)

    def p_act_calculate(self,t):
        """Calculate the activation probability based on antigen exposure and feedback from effector cells"""

        #Antigen exposure
        antigen=self.A_peak*np.exp(-0.5*((t-self.A_tpeak)/self.A_tsigma)**2)

        #Feedback
        n_cells=self.data[-1].copy()
        n_effector=n_cells["TSLEC_response"]+n_cells["Tem_response"]+n_cells["Temra_response"]
        feedback=1/(1+self.feedback_c*n_effector)

        #Activation chance
        p_act=feedback*antigen

        return p_act
        
    def step(self,t):
        """Perform one time step of the simulation, updating the state of each cell based on rules (i.e. probabilities)"""
        #Activation chance
        p_act=self.p_act_calculate(t)

        aliveCells=[]
        newCells=[]
        for cell in self.cells[:]:
            alive=True
            rand=random.random()
            #Only allow one change of cell type per iteration
            if cell.state=="N":
                #DEATH and ACTIVATION are exclusive, so only one can happen
                if rand<self.d_N:
                    alive=False
                elif rand<(self.d_N+p_act):
                    if random.random()<self.slec_fraction:
                        #SLEC path (effector)
                        cell.state="SLEC"

                        #Clonal expansion
                        n_daughters=2**self.b_SLEC-1 
                        daughter_cells=[CD8Cell(state="SLEC") for i in range(n_daughters)]
                        newCells=newCells+daughter_cells
                    else:
                        #MPEC path (memory)
                        cell.state="MPEC"

                        #Clonal expansion
                        n_daughters=2**self.b_MPEC-1
                        daughter_cells=[CD8Cell(state="MPEC") for i in range(n_daughters)]
                        newCells=newCells+daughter_cells

            elif cell.state=="MPEC":
                #Death & differentiation
                if rand<self.d_MPEC:
                    alive=False
                elif rand<(self.d_MPEC+self.f_S):
                    cell.state="S"
                elif rand<(self.d_MPEC+self.f_S+self.f_C):
                    cell.state="C"

            elif cell.state=="SLEC":
                #Death & differentiation
                if rand<self.d_SLEC:
                    alive=False
                elif rand<(self.d_SLEC+self.f_E):
                    cell.state="E"
                elif rand<(self.d_SLEC+self.f_E+self.f_R):
                    cell.state="R"

            elif cell.state=="S":
                #DEATH
                if rand<self.d_S:
                    alive=False
            
            elif cell.state=="C":
                #DEATH
                if rand<self.d_C:
                    alive=False

            elif cell.state=="E":
                #DEATH
                if rand<self.d_E:
                    alive=False

            elif cell.state=="R":
                #DEATH
                if rand<self.d_R:
                    alive=False

            if alive==True:
                aliveCells.append(cell)

        self.cells=aliveCells+newCells

    def getData(self,t,run):
        """Get amount of each cell type and store in self.data"""

        #Get amount of each type and add to dictionary
        N_amount=0
        MPEC_amount=0
        SLEC_amount=0
        S_amount=0
        C_amount=0
        E_amount=0
        R_amount=0

        for cell in self.cells:
            if cell.state=="N":
                N_amount+=1
            elif cell.state=="MPEC":
                MPEC_amount+=1
            elif cell.state=="SLEC":
                SLEC_amount+=1
            elif cell.state=="S":
                S_amount+=1
            elif cell.state=="C":
                C_amount+=1
            elif cell.state=="E":
                E_amount+=1
            elif cell.state=="R":
                R_amount+=1

        self.data.append({
            "Run":run,
            "Time_Point":t,
            "Tnaive_response":N_amount,
            "TMPEC_response":MPEC_amount,
            "TSLEC_response":SLEC_amount,
            "TSCM_response":S_amount,
            "Tcm_response":C_amount,
            "Tem_response":E_amount,
            "Temra_response":R_amount
        })

    def simulate(self,days=365,run=1):
        """Single simulation for given amount of days and given identification (this identifies the random-seed)"""

        # Set seed for reproducibility
        random.seed(run)
        t=0

        # Create naive cells in begin
        self.cells=[CD8Cell() for i in range(self.N_begin)]
        self.data=[]

        self.getData(t,run)

        # Runs steps and save data
        for t in range(1,days+1):
            self.step(t)
            self.getData(t,run)

        return self.data

    def simulateMultiple(self,days=365,amount=50):
        """Running multiple simulations with different seeds for better statistics"""
        dataTotal=[]
        for run in range(1,amount+1):
            dataRun=self.simulate(days=days,run=run)
            dataTotal=dataTotal+dataRun
        dataTotal=pd.DataFrame(dataTotal)
        return dataTotal
    
    #MULTIPROCESSING: same principle as above, but using multiprocessing to speed up simulations
    def simulate_MP(self,args):
        days,run=args
        random.seed(run)
        t=0

        self.cells=[CD8Cell() for i in range(self.N_begin)]
        self.data=[]

        self.getData(t,run)

        for t in range(1,days+1):
            self.step(t)
            self.getData(t,run)

        return self.data

    def simulateMultiple_MP(self,days=365,amount=50):
        params=[(days,i) for i in range(1,amount+1)]

        with mp.Pool() as pool:
            try:
                results = pool.map(self.simulate_MP,params)
            except KeyboardInterrupt: #Safety mechanism
                print("Aborting…")
                pool.terminate()
                pool.join()

        dataTotal=[d for sublist in results for d in sublist]

        return pd.DataFrame(dataTotal)


# NOTE: Models v2, v3 and v4 were developed for exploration. These are deleted as they were not based on biological principles


class Model_ABM_v5():
    """ FDD model"""
    def __init__(self,
                 N_begin=1000,
                 A_tpeak=14,A_tsigma=5,A_peak=0.15,
                 slec_fraction=0.4,
                 feedback_c=0.005,
                 b_MPEC=2,b_SLEC=5,
                 d_N=0.0003, d_MPEC=0.02, d_SLEC=0.05, d_S=0.0002, d_C=0.004, d_E=0.01, d_R=0.02,
                 f_S=0.03,f_C=0.05,f_E=0.06,f_R=0.015
                 ):
        """
        Docstring for __init__
        :param N_begin: Initial number of naive cells
        :param A_tpeak: Time of peak antigen exposure
        :param A_tsigma: Spread of antigen exposure over time
        :param A_peak: Maximum antigen exposure
        :param slec_fraction: Fraction of activated cells that become SLEC (effector)
        :param feedback_c: Feedback constant
        :param b_MPEC: Number of divisions for MPEC clonal expansion
        :param b_SLEC: Number of divisions for  SLEC clonal expansion
        :param d_N: Death rate for naive cells
        :param d_MPEC: Death rate for MPEC cells
        :param d_SLEC: Death rate for SLEC cells
        :param d_S: Death rate for Tscm cells
        :param d_C: Death rate for Tcm cells
        :param d_E: Death rate for Tem cells
        :param d_R: Death rate for Temra cells
        :param f_S: Differentiation rate from MPEC to Tscm
        :param f_C: Differentiation rate from MPEC to Tcm
        :param f_E: Differentiation rate from SLEC to Tem
        :param f_R: Differentiation rate from SLEC to Temra
        """
        
        #Settings
        self.N_begin=N_begin

        self.A_tpeak=A_tpeak
        self.A_tsigma=A_tsigma
        self.A_peak=A_peak

        self.slec_fraction=slec_fraction

        self.feedback_c=feedback_c

        self.b_MPEC=b_MPEC
        self.b_SLEC=b_SLEC

        self.d_N=d_N
        self.d_MPEC=d_MPEC
        self.d_SLEC=d_SLEC
        self.d_S=d_S
        self.d_C=d_C
        self.d_E=d_E
        self.d_R=d_R

        self.f_S=f_S
        self.f_C=f_C
        self.f_E=f_E
        self.f_R=f_R

        #Data
        self.cells=[]
        self.data=[]

    #Convert values for b_MPEC and b_SLEC to integers
    @property
    def b_MPEC(self):
        return self._b_MPEC
    @b_MPEC.setter
    def b_MPEC(self,value):
        self._b_MPEC=round(value)
    @property
    def b_SLEC(self):
        return self._b_SLEC
    @b_SLEC.setter
    def b_SLEC(self,value):
        self._b_SLEC=round(value)

    def antigen_calculate(self,t):
        """Calculate antigen presence"""

        #Antigen exposure
        antigen=self.A_peak*np.exp(-0.5*((t-self.A_tpeak)/self.A_tsigma)**2)

        return antigen

    def feedback_calculate(self):
        """Calculate feedback stength for given amount of effector cells"""

        #Feedback
        n_cells=self.data[-1].copy()
        n_effector=n_cells["TSLEC_response"]+n_cells["Tem_response"]+n_cells["Temra_response"]
        feedback_param=self.feedback_c*n_effector

        return feedback_param
        
    def step(self,t):
        """Perform one time step of the simulation, updating the state of each cell based on rules (i.e. probabilities)"""

        #Get antigen and feedback
        antigen=self.antigen_calculate(t)
        feedback_param=self.feedback_calculate()

        #Activation chance
        p_act=antigen

        #Feedback influencing death rate
        feedback_death=feedback_param/(feedback_param+1)

        newCells=[]
        aliveCells=[]
        for cell in self.cells[:]:
            rand=random.random()
            alive=True
            #Only allow one change of cell type per iteration
            if cell.state=="N":
                death=self.d_N #Naive cells not influenced by feedback mechanism
                #DEATH and ACTIVATION are exclusive, so only one can happen
                if rand<death:
                    alive=False
                elif rand<(death+p_act):
                    if random.random()<self.slec_fraction:
                        #SLEC path (effector)
                        cell.state="SLEC"

                        #Clonal expansion
                        n_daughters=2**self.b_SLEC-1 
                        daughter_cells=[CD8Cell(state="SLEC") for i in range(n_daughters)]
                        newCells=newCells+daughter_cells
                    else:
                        #MPEC path (memory)
                        cell.state="MPEC"

                        #Clonal expansion
                        n_daughters=2**self.b_MPEC-1
                        daughter_cells=[CD8Cell(state="MPEC") for i in range(n_daughters)]
                        newCells=newCells+daughter_cells

                
            
            elif cell.state=="MPEC":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_MPEC*(1-feedback_death)+feedback_death
                #Death & differentiation
                if rand<death:
                    alive=False
                elif rand<(death+self.f_S):
                    cell.state="S"
                elif rand<(death+self.f_S+self.f_C):
                    cell.state="C"

                

            elif cell.state=="SLEC":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_SLEC*(1-feedback_death)+feedback_death
                #Death & differentiation
                if rand<death:
                    alive=False
                elif rand<(death+self.f_E):
                    cell.state="E"
                elif rand<(death+self.f_E+self.f_R):
                    cell.state="R"

                

            elif cell.state=="S":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_S*(1-feedback_death)+feedback_death
                #DEATH
                if rand<death:
                    alive=False

                
            
            elif cell.state=="C":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_C*(1-feedback_death)+feedback_death
                #DEATH
                if rand<death:
                    alive=False
                
                

            elif cell.state=="E":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_E*(1-feedback_death)+feedback_death
                #DEATH
                if rand<death:
                    alive=False

                

            elif cell.state=="R":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_R*(1-feedback_death)+feedback_death
                #DEATH
                if rand<death:
                    alive=False


            if alive==True:
                aliveCells.append(cell)

        self.cells=aliveCells+newCells

    def getData(self,t,run):
        """Get amount of each cell type and store in self.data"""

        #Get amount of each type
        N_amount=0
        MPEC_amount=0
        SLEC_amount=0
        S_amount=0
        C_amount=0
        E_amount=0
        R_amount=0

        for cell in self.cells:
            if cell.state=="N":
                N_amount+=1
            elif cell.state=="MPEC":
                MPEC_amount+=1
            elif cell.state=="SLEC":
                SLEC_amount+=1
            elif cell.state=="S":
                S_amount+=1
            elif cell.state=="C":
                C_amount+=1
            elif cell.state=="E":
                E_amount+=1
            elif cell.state=="R":
                R_amount+=1

        self.data.append({
            "Run":run,
            "Time_Point":t,
            "Tnaive_response":N_amount,
            "TMPEC_response":MPEC_amount,
            "TSLEC_response":SLEC_amount,
            "TSCM_response":S_amount,
            "Tcm_response":C_amount,
            "Tem_response":E_amount,
            "Temra_response":R_amount
        })

    def simulate(self,days=365,run=1):
        """Run the simulation for given amount of days and a single run number (for seeding the random generator)"""

        # Set seed for reproducibility
        random.seed(run)
        t=0

        self.cells=[CD8Cell() for i in range(self.N_begin)]
        self.data=[]

        self.getData(t,run)

        for t in range(1,days+1):
            self.step(t)
            self.getData(t,run)

        return self.data


    def simulateMultiple(self,days=365,amount=50):
        """Running multiple simulations with different seeds for better statistics"""
        dataTotal=[]
        for run in range(1,amount+1):
            dataRun=self.simulate(days=days,run=run)
            dataTotal=dataTotal+dataRun
        dataTotal=pd.DataFrame(dataTotal)
        return dataTotal
    
    #MULTIPROCESSING: same principle as above, but using multiprocessing to speed up simulations

    def simulate_MP(self,args):
        days,run=args
        random.seed(run)
        t=0

        self.cells=[CD8Cell() for i in range(self.N_begin)]
        self.data=[]

        self.getData(t,run)

        for t in range(1,days+1):
            self.step(t)
            self.getData(t,run)

        return self.data

    def simulateMultiple_MP(self,days=365,amount=50):
        params=[(days,i) for i in range(1,amount+1)]

        with mp.Pool() as pool:
            try:
                results = pool.map(self.simulate_MP,params)
            except KeyboardInterrupt: #Safety mechanism
                print("Aborting…")
                pool.terminate()
                pool.join()

        dataTotal=[d for sublist in results for d in sublist]

        return pd.DataFrame(dataTotal)

class Model_ABM_v5_2():
    """FDD model + Tem renewal"""
    def __init__(self,
                 N_begin=1000,
                 A_tpeak=14,A_tsigma=5,A_peak=0.3,
                 slec_fraction=0.4,
                 feedback_c=0.005,
                 b_MPEC=2,b_SLEC=5,
                 d_N=0.0003, d_MPEC=0.02, d_SLEC=0.05, d_S=0.0002, d_C=0.004, d_E=0.01, d_R=0.02,
                 f_S=0.03,f_C=0.05,f_E=0.06,f_R=0.015,
                 ren_E=0.005
                 ):
        """
        Docstring for __init__
        :param N_begin: Initial number of naive cells
        :param A_tpeak: Time of peak antigen exposure
        :param A_tsigma: Spread of antigen exposure over time
        :param A_peak: Maximum antigen exposure
        :param slec_fraction: Fraction of activated cells that become SLEC (effector)
        :param feedback_c: Feedback constant
        :param b_MPEC: Number of divisions for MPEC clonal expansion
        :param b_SLEC: Number of divisions for  SLEC clonal expansion
        :param d_N: Death rate for naive cells
        :param d_MPEC: Death rate for MPEC cells
        :param d_SLEC: Death rate for SLEC cells
        :param d_S: Death rate for Tscm cells
        :param d_C: Death rate for Tcm cells
        :param d_E: Death rate for Tem cells
        :param d_R: Death rate for Temra cells
        :param f_S: Differentiation rate from MPEC to Tscm
        :param f_C: Differentiation rate from MPEC to Tcm
        :param f_E: Differentiation rate from SLEC to Tem
        :param f_R: Differentiation rate from SLEC to Temra
        :param ren_E: Renewal rate for Tem cells
        """
        
        #Settings
        self.N_begin=N_begin

        self.A_tpeak=A_tpeak
        self.A_tsigma=A_tsigma
        self.A_peak=A_peak

        self.slec_fraction=slec_fraction

        self.feedback_c=feedback_c

        self.b_MPEC=b_MPEC
        self.b_SLEC=b_SLEC

        self.d_N=d_N
        self.d_MPEC=d_MPEC
        self.d_SLEC=d_SLEC
        self.d_S=d_S
        self.d_C=d_C
        self.d_E=d_E
        self.d_R=d_R

        self.f_S=f_S
        self.f_C=f_C
        self.f_E=f_E
        self.f_R=f_R

        self.ren_E=ren_E

        #Data
        self.cells=[]
        self.data=[]

    #Convert values for b_MPEC and b_SLEC to integers
    @property
    def b_MPEC(self):
        return self._b_MPEC
    @b_MPEC.setter
    def b_MPEC(self,value):
        self._b_MPEC=round(value)
    @property
    def b_SLEC(self):
        return self._b_SLEC
    @b_SLEC.setter
    def b_SLEC(self,value):
        self._b_SLEC=round(value)

    def antigen_calculate(self,t):
        """Calculate antigen presence"""

        #Antigen exposure
        antigen=self.A_peak*np.exp(-0.5*((t-self.A_tpeak)/self.A_tsigma)**2)

        return antigen

    def feedback_calculate(self):
        """Calculate feedback stength for given amount of effector cells"""

        #Feedback
        n_cells=self.data[-1].copy()
        n_effector=n_cells["TSLEC_response"]+n_cells["Tem_response"]+n_cells["Temra_response"]
        feedback_param=self.feedback_c*n_effector

        return feedback_param
        
    def step(self,t):
        """Perform one time step of the simulation, updating the state of each cell based on rules (i.e. probabilities)"""

        #Get antigen and feedback
        antigen=self.antigen_calculate(t)
        feedback_param=self.feedback_calculate()

        #Activation chance
        p_act=antigen

        #Feedback influencing death rate
        feedback_death=feedback_param/(feedback_param+1)

        newCells=[]
        aliveCells=[]
        for cell in self.cells[:]:
            rand=random.random()
            alive=True
            #Only allow one change of cell type per iteration
            if cell.state=="N":
                death=self.d_N #Naive cells not influenced by mechanism
                #DEATH and ACTIVATION are exclusive, so only one can happen
                if rand<death:
                    alive=False
                elif rand<(death+p_act):
                    if random.random()<self.slec_fraction:
                        #SLEC path (effector)
                        cell.state="SLEC"

                        #Clonal expansion
                        n_daughters=2**self.b_SLEC-1 
                        daughter_cells=[CD8Cell(state="SLEC") for i in range(n_daughters)]
                        newCells=newCells+daughter_cells
                    else:
                        #MPEC path (memory)
                        cell.state="MPEC"

                        #Clonal expansion
                        n_daughters=2**self.b_MPEC-1
                        daughter_cells=[CD8Cell(state="MPEC") for i in range(n_daughters)]
                        newCells=newCells+daughter_cells

                
            
            elif cell.state=="MPEC":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_MPEC*(1-feedback_death)+feedback_death
                #Death & differentiation
                if rand<death:
                    alive=False
                elif rand<(death+self.f_S):
                    cell.state="S"
                elif rand<(death+self.f_S+self.f_C):
                    cell.state="C"

                

            elif cell.state=="SLEC":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_SLEC*(1-feedback_death)+feedback_death
                #Death & differentiation
                if rand<death:
                    alive=False
                elif rand<(death+self.f_E):
                    cell.state="E"
                elif rand<(death+self.f_E+self.f_R):
                    cell.state="R"

                

            elif cell.state=="S":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_S*(1-feedback_death)+feedback_death
                #DEATH
                if rand<death:
                    alive=False

                
            
            elif cell.state=="C":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_C*(1-feedback_death)+feedback_death
                #DEATH
                if rand<death:
                    alive=False
                
                

            elif cell.state=="E":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_E*(1-feedback_death)+feedback_death
                #DEATH
                if rand<death:
                    alive=False
                elif rand<(death+self.ren_E):
                    newCells.append(CD8Cell(state="E"))

                

            elif cell.state=="R":
                # Total death rate, taken into account the feedback mechanism
                death=self.d_R*(1-feedback_death)+feedback_death
                #DEATH
                if rand<death:
                    alive=False
                


            if alive==True:
                aliveCells.append(cell)

        self.cells=aliveCells+newCells

    def getData(self,t,run):
        """Get amount of each cell type and store in self.data"""

        #Get amount of each type
        N_amount=0
        MPEC_amount=0
        SLEC_amount=0
        S_amount=0
        C_amount=0
        E_amount=0
        R_amount=0

        for cell in self.cells:
            if cell.state=="N":
                N_amount+=1
            elif cell.state=="MPEC":
                MPEC_amount+=1
            elif cell.state=="SLEC":
                SLEC_amount+=1
            elif cell.state=="S":
                S_amount+=1
            elif cell.state=="C":
                C_amount+=1
            elif cell.state=="E":
                E_amount+=1
            elif cell.state=="R":
                R_amount+=1

        self.data.append({
            "Run":run,
            "Time_Point":t,
            "Tnaive_response":N_amount,
            "TMPEC_response":MPEC_amount,
            "TSLEC_response":SLEC_amount,
            "TSCM_response":S_amount,
            "Tcm_response":C_amount,
            "Tem_response":E_amount,
            "Temra_response":R_amount
        })

    def simulate(self,days=365,run=1):
        """Run the simulation for given amount of days and a single run number (for seeding the random generator)"""

        # Set seed for reproducibility
        random.seed(run)
        t=0

        self.cells=[CD8Cell() for i in range(self.N_begin)]
        self.data=[]

        self.getData(t,run)

        for t in range(1,days+1):
            self.step(t)
            self.getData(t,run)

        return self.data


    #REPLICATION
    def simulateMultiple(self,days=365,amount=50):
        """Running multiple simulations with different seeds for better statistics"""

        dataTotal=[]
        for run in range(1,amount+1):
            dataRun=self.simulate(days=days,run=run)
            dataTotal=dataTotal+dataRun
        dataTotal=pd.DataFrame(dataTotal)
        return dataTotal
    
    #MULTIPROCESSING: same principle as above, but using multiprocessing to speed up simulations
    def simulate_MP(self,args):
        days,run=args
        random.seed(run)
        t=0

        self.cells=[CD8Cell() for i in range(self.N_begin)]
        self.data=[]

        self.getData(t,run)

        for t in range(1,days+1):
            self.step(t)
            self.getData(t,run)

        return self.data

    def simulateMultiple_MP(self,days=365,amount=50):
        params=[(days,i) for i in range(1,amount+1)]

        with mp.Pool() as pool:
            try:
                results = pool.map(self.simulate_MP,params)
            except KeyboardInterrupt: #Safety mechanism
                print("Aborting…")
                pool.terminate()
                pool.join()

        dataTotal=[d for sublist in results for d in sublist]

        return pd.DataFrame(dataTotal)


class Model_ABM_v6():
    """ADDC model"""
    def __init__(self,
                 N_begin=1000,
                 A_tpeak=14,A_tsigma=5,A_peak=0.3,
                 slec_fraction=0.4,
                 feedback_c=0,
                 b_MPEC=2,b_SLEC=5,
                 d_N=0.0003, d_MPEC=0.02, d_SLEC=0.05, d_S=0.0002, d_C=0.004, d_E=0.01, d_R=0.02,
                 f_S=0.03,f_C=0.05,f_E=0.06,f_R=0.015,
                 contraction_c=10,
                 ren_E=0.005
                 ):
        """
        Docstring for __init__
        :param N_begin: Initial number of naive cells
        :param A_tpeak: Time of peak antigen exposure
        :param A_tsigma: Spread of antigen exposure over time
        :param A_peak: Maximum antigen exposure
        :param slec_fraction: Fraction of activated cells that become SLEC (effector)
        :param feedback_c: Feedback constant
        :param b_MPEC: Number of divisions for MPEC clonal expansion
        :param b_SLEC: Number of divisions for  SLEC clonal expansion
        :param d_N: Death rate for naive cells
        :param d_MPEC: Death rate for MPEC cells
        :param d_SLEC: Death rate for SLEC cells
        :param d_S: Death rate for Tscm cells
        :param d_C: Death rate for Tcm cells
        :param d_E: Death rate for Tem cells
        :param d_R: Death rate for Temra cells
        :param f_S: Differentiation rate from MPEC to Tscm
        :param f_C: Differentiation rate from MPEC to Tcm
        :param f_E: Differentiation rate from SLEC to Tem
        :param f_R: Differentiation rate from SLEC to Temra
        :param contraction_c: Contraction constant, influencing how strongly the decline of antigen influences cell death
        :param ren_E: Renewal rate for Tem cells"""
        
        #Settings
        self.N_begin=N_begin

        self.A_tpeak=A_tpeak
        self.A_tsigma=A_tsigma
        self.A_peak=A_peak

        self.slec_fraction=slec_fraction

        self.feedback_c=feedback_c

        self.b_MPEC=b_MPEC
        self.b_SLEC=b_SLEC

        self.d_N=d_N
        self.d_MPEC=d_MPEC
        self.d_SLEC=d_SLEC
        self.d_S=d_S
        self.d_C=d_C
        self.d_E=d_E
        self.d_R=d_R

        self.f_S=f_S
        self.f_C=f_C
        self.f_E=f_E
        self.f_R=f_R

        self.contraction_c=contraction_c

        self.ren_E=ren_E

        #Data
        self.cells=[]
        self.data=[]

        self.antigen=0

    #Convert values for b_MPEC and b_SLEC to integers
    @property
    def b_MPEC(self):
        return self._b_MPEC
    @b_MPEC.setter
    def b_MPEC(self,value):
        self._b_MPEC=round(value)
    @property
    def b_SLEC(self):
        return self._b_SLEC
    @b_SLEC.setter
    def b_SLEC(self,value):
        self._b_SLEC=round(value)

    def antigen_calculate(self,t):
        """Calculate antigen presence"""

        #Antigen exposure
        self.antigen=self.A_peak*np.exp(-0.5*((t-self.A_tpeak)/self.A_tsigma)**2)


    def feedback_calculate(self):
        """Calculate feedback stength for given amount of effector cells"""

        #Feedback
        n_cells=self.data[-1].copy()
        n_effector=n_cells["TSLEC_response"]+n_cells["Tem_response"]+n_cells["Temra_response"]
        feedback_param=self.feedback_c*n_effector

        return feedback_param
        
    def step(self,t):
        """Perform one time step of the simulation, updating the state of each cell based on rules (i.e. probabilities)"""

        
        antigen_previous=self.antigen #Save previous antigen value
        self.antigen_calculate(t) #Updates antigen value

        #Feedback
        feedback_param=self.feedback_calculate()
        #Activation chance
        p_act=self.antigen*1/(1+feedback_param)

        #Contraction
        antigen_decline=max(0,antigen_previous-self.antigen)
        contraction_d=self.contraction_c*antigen_decline



        aliveCells=[]
        newCells=[]
        for cell in self.cells[:]:
            alive=True
            rand=random.random()
            #Only allow one change of cell type per iteration
            if cell.state=="N":
                death=self.d_N
                #DEATH and ACTIVATION are exclusive, so only one can happen
                if rand<death:
                    alive=False
                elif rand<(death+p_act):
                    if random.random()<self.slec_fraction:
                        #SLEC path (effector)
                        cell.state="SLEC"

                        #Clonal expansion
                        n_daughters=2**self.b_SLEC-1 
                        daughter_cells=[CD8Cell(state="SLEC") for i in range(n_daughters)]
                        newCells=newCells+daughter_cells
                    else:
                        #MPEC path (memory)
                        cell.state="MPEC"

                        #Clonal expansion
                        n_daughters=2**self.b_MPEC-1
                        daughter_cells=[CD8Cell(state="MPEC") for i in range(n_daughters)]
                        newCells=newCells+daughter_cells

            elif cell.state=="MPEC":
                # Total death rate, taken into account the contraction mechanism
                death=self.d_MPEC+contraction_d
                #Death & differentiation
                if rand<death:
                    alive=False
                elif rand<(death+self.f_S):
                    cell.state="S"
                elif rand<(death+self.f_S+self.f_C):
                    cell.state="C"

            elif cell.state=="SLEC":
                # Total death rate, taken into account the contraction mechanism
                death=self.d_SLEC+contraction_d
                #Death & differentiation
                if rand<death:
                    alive=False
                elif rand<(death+self.f_E):
                    cell.state="E"
                elif rand<(death+self.f_E+self.f_R):
                    cell.state="R"

            elif cell.state=="S":
                # Total death rate, taken into account the contraction mechanism
                death=self.d_S+contraction_d
                #DEATH
                if rand<death:
                    alive=False
            
            elif cell.state=="C":
                # Total death rate, taken into account the contraction mechanism
                death=self.d_C+contraction_d
                #DEATH
                if rand<death:
                    alive=False

            elif cell.state=="E":
                # Total death rate, taken into account the contraction mechanism
                death=self.d_E+contraction_d
                #DEATH
                if rand<death:
                    alive=False
                elif rand<(death+self.ren_E):
                    newCells.append(CD8Cell(state="E"))

            elif cell.state=="R":
                # Total death rate, taken into account the contraction mechanism
                death=self.d_R+contraction_d
                #DEATH
                if rand<death:
                    alive=False

            if alive==True:
                aliveCells.append(cell)

        self.cells=aliveCells+newCells

    def getData(self,t,run):
        """Get amount of each cell type and store in self.data"""

        #Get amount of each type
        N_amount=0
        MPEC_amount=0
        SLEC_amount=0
        S_amount=0
        C_amount=0
        E_amount=0
        R_amount=0

        for cell in self.cells:
            if cell.state=="N":
                N_amount+=1
            elif cell.state=="MPEC":
                MPEC_amount+=1
            elif cell.state=="SLEC":
                SLEC_amount+=1
            elif cell.state=="S":
                S_amount+=1
            elif cell.state=="C":
                C_amount+=1
            elif cell.state=="E":
                E_amount+=1
            elif cell.state=="R":
                R_amount+=1

        self.data.append({
            "Run":run,
            "Time_Point":t,
            "Tnaive_response":N_amount,
            "TMPEC_response":MPEC_amount,
            "TSLEC_response":SLEC_amount,
            "TSCM_response":S_amount,
            "Tcm_response":C_amount,
            "Tem_response":E_amount,
            "Temra_response":R_amount
        })


    def simulate(self,days=365,run=1):
        """Run the simulation for given amount of days and a single run number (for seeding the random generator)"""

        # Set seed for reproducibility
        random.seed(run)
        t=0

        self.cells=[CD8Cell() for i in range(self.N_begin)]
        self.data=[]

        self.antigen_calculate(t) #Calculate antigen presence at day 0. Necessary to get correct antigen decline at day 1.
        self.getData(t,run)

        for t in range(1,days+1):
            self.step(t)
            self.getData(t,run)

        return self.data


    def simulateMultiple(self,days=365,amount=50):
        """Running multiple simulations with different seeds for better statistics"""

        dataTotal=[]
        for run in range(1,amount+1):
            dataRun=self.simulate(days=days,run=run)
            dataTotal=dataTotal+dataRun
        dataTotal=pd.DataFrame(dataTotal)
        return dataTotal
    

    #MULTIPROCESSING: same principle as above, but using multiprocessing to speed up simulations
    def simulate_MP(self,args):
        days,run=args
        random.seed(run)
        t=0

        self.cells=[CD8Cell() for i in range(self.N_begin)]
        self.data=[]

        self.getData(t,run)

        for t in range(1,days+1):
            self.step(t)
            self.getData(t,run)

        return self.data

    def simulateMultiple_MP(self,days=365,amount=50):
        params=[(days,i) for i in range(1,amount+1)]

        with mp.Pool() as pool:
            try:
                results = pool.map(self.simulate_MP,params)
            except KeyboardInterrupt: #Safety mechanism
                print("Aborting…")
                pool.terminate()
                pool.join()

        dataTotal=[d for sublist in results for d in sublist]

        return pd.DataFrame(dataTotal)

   


if __name__=="__main__":
    TCell_Types_Responding=["TSCM_response","Tcm_response","Tem_response","Temra_response"]

    test_model=Model_ABM()
    data=test_model.simulateMultiple_MP(days=365,amount=50)

    dataMean=data.groupby(["Time_Point"])[TCell_Types_Responding].median()
    dataTimePoints=dataMean.index

    # Create a figure for comparison    
    fig=go.Figure()

    color_map = {
                tcell: color
                for tcell, color in zip(TCell_Types_Responding,
                                        px.colors.qualitative.Safe)
            }

    for TCell_Type in TCell_Types_Responding:
        color = color_map[TCell_Type]
        fig.add_trace(go.Scatter(
            x=dataTimePoints,
            y=dataMean[TCell_Type],
            mode="lines",
            line=dict(color=color),
            name=f"{TCell_Type} model",
            legendgroup=TCell_Type  # IMPORTANT: links band and line
        ))

    fig.show()


    
