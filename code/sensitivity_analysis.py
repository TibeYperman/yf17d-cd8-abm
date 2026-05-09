
#Bachelor theis: file for sensitivity analysis
#Author: Tibe Yperman
#E-mail: tibe.yperman@student.uantwerpen.be
#Last revision: 09/05/2026


# NOTE: Requires experimental data (not yet publicly available)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio



from models_ABM import Model_ABM,Model_ABM_v5,Model_ABM_v5_2,Model_ABM_v6



class SensitivityAnalysis():
    """Class for performing sensitivity analysis on the ABM models. 
    The run method performs the sensitivity analysis for a given parameter and returns a dataframe with the results."""

    def __init__(self,model=Model_ABM,opt_file=pd.DataFrame(),opt_name=""):
        """docstring for __init__:
        :param model: The ABM model to perform sensitivity analysis on.
        :param opt_file: A dataframe containing the optimal parameters for the model.
        :param opt_name: The name of the optimal parameter set to use from the opt_file dataframe."""

        self.model=model
        self.opt_file=opt_file
        self.opt_name=opt_name

        # Amount of seeds to use for each parameter value in the sensitivity analysis.
        self.amountSeeds=500

        #Get experimental stats and put them in same format as the model data
        expStats=pd.read_csv("Files/Stats.csv")
        self.data_exp=expStats[expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False).to_numpy() #Same form as data_model
        self.data_exp_normalization=np.broadcast_to(np.max(self.data_exp,0),(4,4)) #Get normalization based on the maximum experimental values
        self.number_of_datapoints=np.size(self.data_exp)

        self.TCell_Types_Responding=["TSCM_response","Tcm_response","Tem_response","Temra_response"]
    
    def run(self,param=""):
        """Run the sensitivity analysis for a given parameter. This method will vary the parameter in a range around the optimal value and calculate the RMSE for each value. The result is returned in a dataframe."""

        #Get dictionary of optimal parameters
        params_dict=self.opt_file[self.opt_file["Description"]==self.opt_name]["Parameters"].iloc[0]

        #Set the model params to these optimal values
        for i,v in params_dict.items(): #Set variable model parameters inside the model attribute (class)
            setattr(self.model,i,v)

        #Get array of the variable param for sensitivity analysis. We'll check from 0 to 2 times the optimal value (in most cases).
        param_value=params_dict[param]
        param_ranges=np.linspace(param_value*0,param_value*2,num=201,endpoint=True)
        idx=np.argmax(param_ranges>0.01)
        param_range=param_ranges[:idx]
        #param_range=[round(param_value)-1,round(param_value),round(param_value)+1]

        #Get RMSE for each value
        RMSE_range=[]
        for v in param_range:
            setattr(self.model,param,v)

            data=self.model.simulateMultiple_MP(days=365,amount=self.amountSeeds)
            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()

            RMSE=self.loss_calculate(dataMean=dataMean)
            print(RMSE)

            RMSE_range.append(RMSE)

        #Return result
        return pd.DataFrame([{"Parameter":param,"Parameter_range":param_range,"RMSE_range":RMSE_range}])

    def loss_calculate(self,dataMean):
        """Calculate the RMSE between the model data and the experimental data for a given parameter value."""

        #Calculate numerator of formula s
        data_model=dataMean.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
        data_model=data_model.to_numpy()
        product_s=data_model*self.data_exp
        numerator_s=product_s.sum()

        #Calculate denominator
        square_s=data_model**2
        denominator_s=square_s.sum()

        #Calculate s
        s=numerator_s/denominator_s

        #Calculate normalized RMSE
        loss_matrix=((self.data_exp-s*data_model)/self.data_exp_normalization)**2
        SSE=loss_matrix.sum()
        RMSE=np.sqrt(SSE/self.number_of_datapoints)

        return RMSE


def model_v1():
    """Run sensitivity analysis for the first version of the ABM model."""

    # Choose optimization result
    sens=SensitivityAnalysis()
    sens.model=Model_ABM()
    sens.opt_file=pd.read_pickle("Thesis optimization/ABM_v1.pkl")
    sens.opt_name="Final 1 scipy"

    # Choose parameter, run the analysis and add result to pandas dataframe
    df=sens.run(param="slec_fraction")
    df_old=pd.read_pickle("Sensitivity analysis/ABM_v1.pkl")
    df_new=pd.concat([df_old,df],ignore_index=True)

    df_new.to_pickle("Sensitivity analysis/ABM_v1.pkl")

def model_v5():
    """Run sensitivity analysis for v5 of the ABM model."""

    # Choose optimization result
    sens=SensitivityAnalysis()
    sens.model=Model_ABM_v5()
    sens.opt_file=pd.read_pickle("Optimization files/ABM_v5.pkl")
    sens.opt_name="Final 1 scipy"

    # Choose parameter, run the analysis and add result to pandas dataframe
    df=sens.run(param="slec_fraction")
    df_old=pd.read_pickle("Sensitivity analysis/ABM_v5.pkl")
    df_new=pd.concat([df_old,df],ignore_index=True)

    df_new.to_pickle("Sensitivity analysis/ABM_v5.pkl")

def model_v5_2():
    """Run sensitivity analysis for v5.2 of the ABM model."""

    # Choose optimization result
    sens=SensitivityAnalysis()
    sens.model=Model_ABM_v5_2()
    sens.opt_file=pd.read_pickle("Optimization files/ABM_v5_2.pkl")
    sens.opt_name="Final 1 scipy"

    # Choose parameter, run the analysis and add result to pandas dataframe
    df=sens.run(param="ren_E")
    df_old=pd.read_pickle("Sensitivity analysis/ABM_v5_2.pkl")
    df_new=pd.concat([df_old,df],ignore_index=True)

    df_new.to_pickle("Sensitivity analysis/ABM_v5_2.pkl")

def model_v6():
    """Run sensitivity analysis for v6 of the ABM model."""

    # Choose optimization result
    sens=SensitivityAnalysis()
    sens.model=Model_ABM_v6()
    sens.opt_file=pd.read_pickle("Thesis optimization/ABM_v6.pkl")
    sens.opt_name="Final 4 optuna"

    # Choose parameter, run the analysis and add result to pandas dataframe
    df=sens.run(param="ren_E")
    df_old=pd.read_pickle("Sensitivity analysis/ABM_v6.pkl")
    df_new=pd.concat([df_old,df],ignore_index=True)

    df_new.to_pickle("Sensitivity analysis/ABM_v6.pkl")


if __name__=="__main__":
    # Run sensitivity analysis
    #model_v6()


    # Visualization of sensitivity analysis results
    df=pd.read_pickle("Sensitivity analysis/ABM_v6.pkl")
    df=df[df["Parameter"]=="ren_E"]

    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Parameter_range"].iloc[0],
        y=df["RMSE_range"].iloc[0],
        mode="lines+markers",
        showlegend=False,
    ))

    fig.add_trace(
    go.Scatter(
        x=[df["Parameter_range"].iloc[0][-5]],
        y=[df["RMSE_range"].iloc[0][-5]],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="Minimum found",
        hovertemplate="x = %{x}<br>y = %{y}<extra></extra>",
        showlegend=False,
    )
)

    fig.update_layout(
        template="plotly_white",
        title="Sensitivity analysis: ren_E",
        xaxis_title="ren_E",
        yaxis_title="RMSE",
        font=dict(size=24),
    )

    fig.show()

    #Save figure
    pio.write_image(fig, "plot.png", scale=4, width=2000, height=1200)
