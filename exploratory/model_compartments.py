
#Bachelor theis: file containing ODE model
#Author: Tibe Yperman
#E-mail: tibe.yperman@student.uantwerpen.be
#Last revision: 09/05/2026



# NOTE: This file is not used for the thesis, but is an exploratory file. We made a simple ODE model to understand the dynamics of the T cell response. 
# The models used for the thesis are ABMs, which are implemented in the models_ABM.py file. 
# This file is kept for reference and to show the initial ideas for modelling the T cell response.

import pandas as pd

#Visualization
import plotly.graph_objects as go
import plotly.express as px

TCell_Types_All=["Tnaive_response","TSCM_response","Tcm_response","Tem_response","Temra_response"]
TCell_Types_Responding=["TSCM_response","Tcm_response","Tem_response","Temra_response"]

class Model_Compartment():
    """ODE model for the T cell response. The model contains 5 compartments: Tnaive, Tscm, Tcm, Tem and Temra.
    The model is based on the following assumptions:
    - Naive T cells are activated by the presence of antigen, and can differentiate into Tscm or Tem directly
    - Tscm can differentiate into Tcm, which can differentiate into Tem, which can differentiate into Temra
    - Proliferation of Tscm, Tcm and Tem is only present when antigen is present
    - Death is present in all compartments, but is higher in the more differentiated compartments.
    In this model there is no use of MPECs or SLECs"""
    def __init__(self,
                 N_begin=10**6,A_begin=1,
                 A_duration=14,
                 p_act=0.0001,frac_S=0.3,
                 p_SC=0.05,p_CE=0.05,p_ER=0.10,
                 r_S=0.02, r_C=0.05, r_E=0.5,
                 d_N=0.0005, d_S=0.001, d_C=0.002, d_E=0.3, d_R=0.15,
                 ):
        """
        Docstring for __init__
        
        :param N_begin: Amount of initial naive T cells
        :param A_amount: Amount of antigen
        :param A_duration: Duration of antigen presence
        :param p_act: Fraction of naive T cells activated when antigen present
        :param frac_S: Fraction of memory biased activation
        :param p_SC: Fraction of Tscm differentiating into Tcm
        :param p_CE: Fraction of Tcm differentiating into Tem
        :param p_ER: Fraction of Tem differentiating into Temra
        :param r_S: Proliferation rate of Tscm when antigen present
        :param r_C: Proliferation rate of Tcm when antigen present
        :param r_E: Proliferation rate of Tem when antigen present
        :param d_N: Death rate of Tnaive
        :param d_S: Death rate of Tscm
        :param d_C: Death rate of Tcm
        :param d_E: Death rate of Tem
        :param d_R: Death rate of Temra
        """
        #Beginning amount of cells
        self.N_begin=N_begin
        self.A_begin=A_begin

        #Amounts of cells
        self.N=self.N_begin
        self.S=0
        self.C=0
        self.E=0
        self.R=0
        self.A=self.A_begin

        #Antigen duration
        self.A_duration=A_duration

        #Naive activation
        self.p_act=p_act
        self.frac_S=frac_S

        #Differentiation
        self.p_SC=p_SC
        self.p_CE=p_CE
        self.p_ER=p_ER

        #Proliferation when antigen present
        self.r_S=r_S
        self.r_C=r_C
        self.r_E=r_E

        #Death rates
        self.d_N=d_N
        self.d_S=d_S
        self.d_C=d_C
        self.d_E=d_E
        self.d_R=d_R

    def antigen(self,t):
        """Antigen is implemented as a step function."""
        if t>self.A_duration:
            self.A=0
    
    def changeSubsets(self):
        #Activation + bias
        activated=self.A*self.p_act*self.N
        to_S0=self.frac_S*activated
        to_E0=(1-self.frac_S)*activated

        #Differentiation
        to_C_from_S=self.p_SC*self.S
        to_E_from_C=self.p_CE*self.C
        to_R_from_E=self.p_ER*self.E

        #Proliferation (when antigen present)
        prolif_S=self.A*self.r_S*self.S
        prolif_C=self.A*self.r_C*self.C
        prolif_E=self.A*self.r_E*self.E

        #Death
        death_N=self.d_N*self.N
        death_S=self.d_S*self.S
        death_C=self.d_C*self.C
        death_E=self.d_E*self.E
        death_R=self.d_R*self.R

        #Changes
        dN=-activated-death_N
        dS=to_S0+prolif_S-to_C_from_S-death_S
        dC=to_C_from_S+prolif_C-to_E_from_C-death_C
        dE=to_E0+to_E_from_C+prolif_E-to_R_from_E-death_E
        dR=to_R_from_E-death_R

        #New subsets
        self.N+=dN
        self.S+=dS
        self.C+=dC
        self.E+=dE
        self.R+=dR

    def runSimulation(self,days=365):
        """Run the simulation for a given amount of days."""
        data=[]
        t=0
        self.N=self.N_begin
        self.S=0
        self.C=0
        self.E=0
        self.R=0
        self.A=self.A_begin
        data.append({
            "Time_Point":t,
            "Tnaive_response":self.N,
            "TSCM_response":self.S,
            "Tcm_response":self.C,
            "Tem_response":self.E,
            "Temra_response":self.R
        })
        for t in range(1,days+1):
            self.antigen(t)
            self.changeSubsets()
            data.append({
                "Time_Point":t,
                "Tnaive_response":self.N,
                "TSCM_response":self.S,
                "Tcm_response":self.C,
                "Tem_response":self.E,
                "Temra_response":self.R
            })
    
        # Return the result in pandas dataframe
        data=pd.DataFrame(data)
        return data


def getStandardModel():
    """Example settings of the ODE model"""

    #Amount of cells
    N_begin=10**6
    A_begin=1
    #Duration of antigen presence
    A_duration=14
    #Activation
    p_act=0.0001
    frac_S=0.3
    #Differentiation
    p_SC=0.05
    p_CE=0.05
    p_ER=0.10
    #Profileration
    r_S=0.02
    r_C=0.05 
    r_E=0.5
    #Death rates
    d_N=0.0005 
    d_S=0.001 
    d_C=0.002 
    d_E=0.3
    d_R=0.15

    # Create model with the given settings.
    model=Model_Compartment(N_begin=N_begin,A_begin=A_begin,
                 A_duration=A_duration,
                 p_act=p_act,frac_S=frac_S,
                 p_SC=p_SC,p_CE=p_CE,p_ER=p_ER,
                 r_S=r_S, r_C=r_C, r_E=r_E,
                 d_N=d_N, d_S=d_S, d_C=d_C, d_E=d_E, d_R=d_R)
    return model



if __name__=="__main__":
    
    # Standard model with changed settings
    model=getStandardModel()
    model.r_E=0.005
    model.d_E=0.3
    model.d_S=0.00005
    model.d_C=0.0005
    model.p_act=0.001
    model.A_duration=14

    # Get model result
    data=model.runSimulation()

    # Data normalization for comparison (NOTE: this does not use the correct global scaling factor method)
    sums=data[TCell_Types_Responding].sum(axis=1)
    max_sum=sums.max()
    data[TCell_Types_Responding]=data[TCell_Types_Responding]/max_sum
    data=data.drop(columns="Tnaive_response")

    # NOTE: Experimental data not yet published
    #expStats=pd.read_csv("Files/Stats.csv")

    # Create a figure
    fig=go.Figure()

    color_map = {
                tcell: color
                for tcell, color in zip(TCell_Types_Responding,
                                        px.colors.qualitative.Safe)
            }
    
    for TCell_Type in TCell_Types_Responding:
        # NOTE: Experimental data not yet published
        #data_exp_sub=expStats[expStats["TCell_Type"]==TCell_Type]
        color = color_map[TCell_Type]
        #Get model
        fig.add_trace(go.Scatter(
            x=data["Time_Point"],
            y=data[TCell_Type],
            mode="lines",
            line=dict(color=color),
            name=f"{TCell_Type} model",
            legendgroup=TCell_Type  # IMPORTANT: links band and line
        ))
        # NOTE: Experimental data not yet published
        # fig.add_trace(go.Scatter(
        #     x=data_exp_sub["Time_Point"],
        #     y=data_exp_sub["Median"],
        #     mode="markers",
        #     marker=dict(size=10,color=color),
        #     name=f"{TCell_Type} exp",
        #     legendgroup=TCell_Type  # IMPORTANT: links band and line
        # ))


    fig.show()