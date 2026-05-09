
#Bachelor theis: file used for optimization
#Author: Tibe Yperman
#E-mail: tibe.yperman@student.uantwerpen.be
#Last revision: 09/05/2026

import pandas as pd
import numpy as np
import scipy.optimize
import optuna
import os


from models_ABM import Model_ABM,Model_ABM_v5,Model_ABM_v5_2,Model_ABM_v6



# NOTE: the minimisation classes require the experimental data for fitting (not yet available)
class Minimize_scipy():
    """Minimization of a certain model using Scipy"""
    def __init__(self,model=Model_ABM,amountOfRuns=1,fixed_params={},var_params={},var_params_bounds={}):
        """
            :param model: Which model to optimize
            :param amountOfRuns: Amount of runs that is averaged for each parameter set tried in optimization
            :param fixed_params: dict of fixed parameters
            :param var_params: dict of variable parameters that should be optimized
            :param var_params_bounds: dict of range of variable parameters: (begin, end)
        """

        # Get experimental data for calculating the normalized RMSE
        expStats=pd.read_csv("Files/Stats.csv")
        self.data_exp=expStats[expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False).to_numpy() #Same form as data_model
        self.data_exp_normalization=np.broadcast_to(np.max(self.data_exp,0),(4,4)) #Get normalization based on the maximum experimental values
        self.number_of_datapoints=np.size(self.data_exp)

        self.TCell_Types_All=["Tnaive_response","TSCM_response","Tcm_response","Tem_response","Temra_response"]
        self.TCell_Types_Responding=["TSCM_response","Tcm_response","Tem_response","Temra_response"]

        # Model to minimize
        self.model=model

        #Settings
        self.amountOfRuns=amountOfRuns
        self.fixed_params=fixed_params
        self.var_params=var_params
        self.var_params_bounds=var_params_bounds

        #Data
        self.x0=[] #Starting guess (calculated from var_params)
        self.bounds_list=[]
        self.current_var_params={}
        self.s=0
        self.loss_s=1
        self.dataMean=pd.DataFrame()

        #Minimum
        self.loss_s_min=99
        self.s_min=0
        self.total_params_min={}
        self.dataMean_min=pd.DataFrame()
        

    def model_calculate(self):
        """Get model results"""

        data=self.model.simulateMultiple_MP(days=365,amount=self.amountOfRuns)

        # Get median
        self.dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()

        return

    def loss_calculate(self,):
        """Calculation of the global scaling factor and normalized RMSE"""

        #Calculate numerator of s formula
        data_model=self.dataMean.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
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

        self.s=s

        return RMSE
    
    def convert_parameters(self):
        """Function that sets the fixed model parameters and converts the variable parameters and their boundaries to the list format that is used in the other functions"""

        #Set model parameters
        self.set_model_fixed_parameters()
    
        #Convert variable parameters to list (in the order of the dictionary)
        self.x0=[val for val in self.var_params.values()]

        #Get list of bounds
        bounds=[]
        for key in self.var_params.keys():
            if key in self.var_params_bounds.keys():
                bounds.append(self.var_params_bounds[key])
            else:
                if key=="N_begin":
                    bounds.append((0,None))
                elif key=="A_duration":
                    bounds.append((0,None))
                else:
                    bounds.append((0,1))
        self.bounds_list=bounds

        return

    def set_model_var_parameters(self,variables):
        """Set variable parameters in the model (run each time for another parameter set)"""

        for i,v in variables.items(): #Set variable model parameters inside the model attribute (class)
            setattr(self.model,i,v)

        return
    
    def set_model_fixed_parameters(self):
        """Sets the fixed parameters (called once)"""

        for i,v in self.fixed_params.items(): #Set fixed model parameters inside the model attribute (class)
            setattr(self.model,i,v)

        return

    def minimize_fun(self,x):
        """Function that returns the RMSE for a given variable parameterset. This result is used by the optimizer to find the best solution"""

        #Convert list to corresponding parameters
        self.current_var_params=dict(zip(self.var_params.keys(),x))

        #Set variable model parameters
        self.set_model_var_parameters(self.current_var_params)

        #Calculate model
        self.model_calculate()

        #Get RMSE
        loss_s=self.loss_calculate()

        #Update minimal values
        if loss_s<self.loss_s_min:
            self.loss_s_min=loss_s
            self.s_min=self.s
            self.total_params_min=(self.fixed_params|self.current_var_params).copy()
            self.dataMean_min=self.dataMean.copy()

        print(f"{loss_s:.5g}")

        # Return the RMSE to the optimizer
        return loss_s


    def run(self,descr):
        """Function that calls the optimizer and returns its found minimal solution"""

        # Set fixed model parameters and convert the variable ones to the correct format
        self.convert_parameters()

        # Call the optimizer
        res=scipy.optimize.differential_evolution(self.minimize_fun,bounds=self.bounds_list,maxiter=100, popsize=15, tol=0.001, seed=42)

        # The minimal solution is saved in a pandas dataframe
        solution=[
            {
                "Description":descr,
                "Parameters":self.total_params_min,
                "s":self.s_min,
                "loss_s":self.loss_s_min,
                "Mean data":self.dataMean_min
            },
        ]

        solution=pd.DataFrame(solution)

        return solution

class Minimize_optuna():
    """Minimization of a certain model using Optuna"""
    def __init__(self,model=Model_ABM,amountOfRuns=1,fixed_params={},var_params={},var_params_bounds={}):
        """
            :param model: Which model to optimize
            :param amountOfRuns: Amount of runs that is averaged for each parameter set tried in optimization
            :param fixed_params: dict of fixed parameters with values
            :param var_params: dict of variable parameters names with types that should be optimized
            :param var_params_bounds: dict of range of variable parameters: [begin, end]
        """

        # Get experimental data for calculating the normalized RMSE
        expStats=pd.read_csv("Files/Stats.csv")
        self.data_exp=expStats[expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False).to_numpy() #Same form as data_model
        self.data_exp_normalization=np.broadcast_to(np.max(self.data_exp,0),(4,4)) #Get normalization based on the maximum experimental values
        self.number_of_datapoints=np.size(self.data_exp)

        self.TCell_Types_All=["Tnaive_response","TSCM_response","Tcm_response","Tem_response","Temra_response"]
        self.TCell_Types_Responding=["TSCM_response","Tcm_response","Tem_response","Temra_response"]

        # Model to minimize
        self.model=model

        #Settings
        self.amountOfRuns=amountOfRuns
        self.fixed_params=fixed_params
        self.var_params=var_params
        self.var_params_bounds=var_params_bounds

        #Data
        self.bounds_dict={}
        self.current_var_params={}
        self.s=0
        self.loss_s=1
        self.dataMean=pd.DataFrame()

        #Minimum
        self.loss_s_min=99
        self.s_min=0
        self.total_params_min={}
        self.dataMean_min=pd.DataFrame()
        

    def model_calculate(self):
        """Get model results for the current variable parameters"""

        # Calculate model
        data=self.model.simulateMultiple_MP(days=365,amount=self.amountOfRuns)

        # Get medians
        self.dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()

        return

    def loss_calculate(self,):
        """Calculation of the global scaling factor and normalized RMSE"""

        #Calculate numerator of s formula
        data_model=self.dataMean.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
        data_model=data_model.to_numpy()
        product_s=data_model*self.data_exp
        numerator_s=product_s.sum()

        #Calculate denominator
        square_s=data_model**2
        denominator_s=max(square_s.sum(),0.00001)

        #Calculate s
        s=numerator_s/denominator_s

        #Calculate normalized RMSE
        loss_matrix=((self.data_exp-s*data_model)/self.data_exp_normalization)**2
        SSE=loss_matrix.sum()
        RMSE=np.sqrt(SSE/self.number_of_datapoints)

        self.s=s

        return RMSE
    
    def convert_parameters(self):
        """Function that sets the fixed model parameters and converts the variable parameters and their boundaries to the format that is used in the other functions"""

        #Set fixed model parameters
        self.set_model_fixed_parameters()
    
        #Get dict of all bounds
        bounds={}
        for key in self.var_params.keys():
            if key in self.var_params_bounds.keys():
                bounds[key]=self.var_params_bounds[key]
            else:
                if key=="N_begin":
                    bounds[key]=[0,None]
                elif key=="A_duration":
                    bounds[key]=[0,None]
                else:
                    bounds[key]=[0,1]
        self.bounds_dict=bounds

        return

    def set_model_var_parameters(self,variables):
        """Set variable parameters in the model (run each time for another parameter set)"""

        for i,v in variables.items(): #Set variable model parameters inside the model attribute (class)
            setattr(self.model,i,v)

        return
    
    def set_model_fixed_parameters(self):
        """Sets the fixed parameters (called once)"""

        for i,v in self.fixed_params.items(): #Set fixed model parameters inside the model attribute (class)
            setattr(self.model,i,v)

        return

    def minimize_fun(self,x):
        """Function that returns the RMSE for a given variable parameterset. This result is used by the optimizer to find the best solution"""

        #Convert list to corresponding parameters
        self.current_var_params=x.copy()

        #Set variable model parameters
        self.set_model_var_parameters(self.current_var_params)

        #Calculate model
        self.model_calculate()

        #Get normalized RMSE
        loss_s=self.loss_calculate()

        #Update minimal values
        if loss_s<self.loss_s_min:
            self.loss_s_min=loss_s
            self.s_min=self.s
            self.total_params_min=(self.fixed_params|self.current_var_params).copy()
            self.dataMean_min=self.dataMean.copy()

        # Return the normalized RMSE
        return loss_s
    
    def objective(self,trial):
        """Function that is called by the Optuna optimizer for each parameter set it tries. It returns the RMSE for that parameter set, which is used by the optimizer to find the best solution"""
        
        #Get optuna parameters
        params={}
        for i,v in self.var_params.items():
            bound_begin=self.bounds_dict[i][0]
            bound_end=self.bounds_dict[i][1]
            if v=="float":
                params[i]=trial.suggest_float(i,bound_begin,bound_end)
            elif v=="int":
                params[i]=trial.suggest_int(i,bound_begin,bound_end)

        # Get the normalized RMSE for these parameters and return to Optuna
        return self.minimize_fun(params)




    def run(self,descr="",n_trials=2000):
        """Function that calls the optimizer and returns its found minimal solution"""

        #Set fixed parameters and get bounds
        self.convert_parameters()

        #Create optuna study
        study=optuna.create_study(direction="minimize")
        study.optimize(self.objective,n_trials=n_trials)


        # Save the found optimal solution in a pandas dataframe
        solution=[
            {
                "Description":descr,
                "Parameters":self.total_params_min,
                "s":self.s_min,
                "loss_s":self.loss_s_min,
                "Mean data":self.dataMean_min
            },
        ]

        solution=pd.DataFrame(solution)

        return solution



class Thesis_optimisations():
    """Class that contains the different optimizations that are run for the thesis. Each optimization is a function that calls the Minimize_scipy or Minimize_optuna class with the correct settings for that optimization"""
    def __init__(self):
        pass

    # Check if optimisation exist (and create them if not)
    if not os.path.exists("Thesis_optimisation"):
        os.makedirs("Thesis_optimisation")
    if not os.path.exists("Thesis_optimisation/ABM_v1.pkl"):
        df=pd.DataFrame()
        df.to_pickle("Thesis_optimisation/ABM_v1.pkl")
    if not os.path.exists("Thesis_optimisation/ABM_v5.pkl"):
        df=pd.DataFrame()
        df.to_pickle("Thesis_optimisation/ABM_v5.pkl")
    if not os.path.exists("Thesis_optimisation/ABM_v6.pkl"):
        df=pd.DataFrame()
        df.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    """
        THESIS TESTS: free parameters

        v1
            Final 1 scipy: A_peak[0.01,0.3], feedback_c[0,0.01], slec_fraction[0,1], b_SLEC[3,8]
            Final 1 optuna: A_peak[0.01,0.3], feedback_c[0,0.01], slec_fraction[0,1], b_SLEC[3,8]

        v5
            Final 1 scipy: A_peak[0.01,0.3], feedback_c[0,0.01], slec_fraction[0,1], b_SLEC[3,8]
            Final 1 optuna: A_peak[0.01,0.3], feedback_c[0,0.01], slec_fraction[0,1], b_SLEC[3,8]

        v5_2
            Final 1 scipy: feedback_c[0,0.01], slec_fraction[0,1], b_SLEC[2,8], ren_E[0,0.01]
            Final 1 optuna: feedback_c[0,0.01], slec_fraction[0,1], b_SLEC[2,8], ren_E[0,0.01]
            

        v6
            Final 1 scipy: slec_fraction[0,1], b_SLEC[3,8], contraction_c[1,20], ren_E[0,0.01]
            Final 2 scipy: slec_fraction[0,1], b_SLEC[2,6], contraction_c[1,20], ren_E[0,0.01]
            Final 3 scipy: slec_fraction[0,1], b_SLEC[2,6], contraction_c[1,20], ren_E[0,0.01], A_tpeak[0,20], A_tsigma[1,10]

            Final 1 optuna: slec_fraction[0,1], b_SLEC[3,8], contraction_c[1,20], ren_E[0,0.01]
            Final 2 optuna: slec_fraction[0,1], b_SLEC[2,6], contraction_c[1,20], ren_E[0,0.01]
            Final 3 optuna: slec_fraction[0,1], b_SLEC[2,6], contraction_c[1,20], ren_E[0,0.01], A_tpeak[0,20], A_tsigma[1,10]

            Final 4 optuna: slec_fraction[0,1], b_SLEC[2,6], contraction_c[1,10], ren_E[0,0.01]
            Final 5 optuna: slec_fraction[0,1], b_SLEC[2,6], contraction_c[1,10], ren_E[0,0.01], A_tpeak[0,20], A_tsigma[1,10]
            Final 6 optuna: slec_fraction[0,1], b_SLEC[2,6], contraction_c[1,10], ren_E[0,0.01], A_tpeak[0,20]
    """

    
    def ABM_v1_scipy_1(self):
        descr="Final 1 scipy"
        model=Model_ABM()
        amountOfRuns=50
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,

            #"slec_fraction":0.6,

            "b_MPEC":2,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015
        }
        var_params={
            "A_peak":"float",
            "feedback_c":"float",
            "b_SLEC":"int",
            "slec_fraction":"float",
        }
        var_params_bounds={
            "A_peak":[0.01,0.3],
            "feedback_c":[0,0.01],
            "b_SLEC":[3,8],
            "slec_fraction":[0,1],
        }


        opt_obj=Minimize_scipy(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr)

        # sol=sol[sol["Description"]==f"{descr} minimal"].reset_index(drop=True)

        old=pd.read_pickle("Thesis_optimisation/ABM_v1.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v1.pkl")

    def ABM_v1_optuna_1(self):
        descr="Final 1 optuna"
        model=Model_ABM()
        amountOfRuns=50
        n_trials=2000
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,

            #"slec_fraction":0.6,

            "b_MPEC":2,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015
        }
        var_params={
            "A_peak":"float",
            "feedback_c":"float",
            "b_SLEC":"int",
            "slec_fraction":"float",
        }
        var_params_bounds={
            "A_peak":[0.01,0.3],
            "feedback_c":[0,0.01],
            "b_SLEC":[3,8],
            "slec_fraction":[0,1],
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        # sol=sol[sol["Description"]==f"{descr} minimal"].reset_index(drop=True)

        old=pd.read_pickle("Thesis_optimisation/ABM_v1.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v1.pkl")


    def ABM_v5_scipy_1(self):
        descr="Final 1 scipy"
        model=Model_ABM_v5()
        amountOfRuns=50
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,

            #"slec_fraction":0.6,

            "b_MPEC":2,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015
        }
        var_params={
            "A_peak":"float",
            "feedback_c":"float",
            "b_SLEC":"int",
            "slec_fraction":"float",
        }
        var_params_bounds={
            "A_peak":[0.01,0.3],
            "feedback_c":[0,0.01],
            "b_SLEC":[3,8],
            "slec_fraction":[0,1],
        }


        opt_obj=Minimize_scipy(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr)

        # sol=sol[sol["Description"]==f"{descr} minimal"].reset_index(drop=True)

        old=pd.read_pickle("Thesis_optimisation/ABM_v5.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v5.pkl")

    def ABM_v5_optuna_1(self):
        descr="Final 1 optuna"
        model=Model_ABM_v5()
        amountOfRuns=50
        n_trials=2000
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,

            #"slec_fraction":0.6,

            "b_MPEC":2,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015
        }
        var_params={
            "A_peak":"float",
            "feedback_c":"float",
            "b_SLEC":"int",
            "slec_fraction":"float",
        }
        var_params_bounds={
            "A_peak":[0.01,0.3],
            "feedback_c":[0,0.01],
            "b_SLEC":[3,8],
            "slec_fraction":[0,1],
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        # sol=sol[sol["Description"]==f"{descr} minimal"].reset_index(drop=True)

        old=pd.read_pickle("Thesis_optimisation/ABM_v5.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v5.pkl")


    def ABM_v5_2_scipy_1(self):
        descr="Final 1 scipy"
        model=Model_ABM_v5_2()
        amountOfRuns=50
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            #"slec_fraction":0.6,

            "b_MPEC":2,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            #"A_peak":"float",
            "feedback_c":"float",
            "b_SLEC":"int",
            "slec_fraction":"float",
            "ren_E":"float",
        }
        var_params_bounds={
            #"A_peak":[0.01,0.3],
            "feedback_c":[0,0.01],
            "b_SLEC":[2,8],
            "slec_fraction":[0,1],
            "ren_E":[0,0.01],
        }


        opt_obj=Minimize_scipy(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr)

        # sol=sol[sol["Description"]==f"{descr} minimal"].reset_index(drop=True)

        old=pd.read_pickle("Thesis_optimisation/ABM_v5_2.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v5_2.pkl")

    def ABM_v5_2_optuna_1(self):
        descr="Final 1 optuna"
        model=Model_ABM_v5_2()
        amountOfRuns=50
        n_trials=2000
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            #"slec_fraction":0.6,

            "b_MPEC":2,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            #"A_peak":"float",
            "feedback_c":"float",
            "b_SLEC":"int",
            "slec_fraction":"float",
            "ren_E":"float",
        }
        var_params_bounds={
            #"A_peak":[0.01,0.3],
            "feedback_c":[0,0.01],
            "b_SLEC":[2,8],
            "slec_fraction":[0,1],
            "ren_E":[0,0.01],
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        # sol=sol[sol["Description"]==f"{descr} minimal"].reset_index(drop=True)

        old=pd.read_pickle("Thesis_optimisation/ABM_v5_2.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v5_2.pkl")


    def ABM_v6_Final_1_scipy(self):
        descr="Final 1 scipy"
        model=Model_ABM_v6()
        amountOfRuns=50
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            #"A_tpeak":"int",
            #"A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,20],
            "b_SLEC":[3,8],
            "ren_E":[0,0.01],
            #"A_tpeak":[0,30],
            #"A_tsigma":[1,10]
        }


        opt_obj=Minimize_scipy(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    def ABM_v6_Final_2_scipy(self):
        descr="Final 2 scipy"
        model=Model_ABM_v6()
        amountOfRuns=50
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            #"A_tpeak":"int",
            #"A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,20],
            "b_SLEC":[2,6],
            "ren_E":[0,0.01],
            #"A_tpeak":[0,30],
            #"A_tsigma":[1,10]
        }


        opt_obj=Minimize_scipy(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    def ABM_v6_Final_3_scipy(self):
        descr="Final 3 scipy"
        model=Model_ABM_v6()
        amountOfRuns=50
        fixed_params={
            "N_begin":1000,

            #"A_tpeak":14,
            #"A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            "A_tpeak":"int",
            "A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,20],
            "b_SLEC":[2,6],
            "ren_E":[0,0.01],
            "A_tpeak":[0,20],
            "A_tsigma":[1,10]
        }


        opt_obj=Minimize_scipy(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    def ABM_v6_Final_4_scipy(self):
        descr="Final 4 scipy"
        model=Model_ABM_v6()
        amountOfRuns=50
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            #"A_tpeak":"int",
            #"A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,10],
            "b_SLEC":[2,6],
            "ren_E":[0,0.01],
            #"A_tpeak":[0,30],
            #"A_tsigma":[1,10]
        }


        opt_obj=Minimize_scipy(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")


    def ABM_v6_Final_1_optuna(self):
        descr="Final 1 optuna"
        model=Model_ABM_v6()
        amountOfRuns=50
        n_trials=2000
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            #"A_tpeak":"int",
            #"A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,20],
            "b_SLEC":[3,8],
            "ren_E":[0,0.01],
            #"A_tpeak":[0,30],
            #"A_tsigma":[1,10]
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    def ABM_v6_Final_2_optuna(self):
        descr="Final 2 optuna"
        model=Model_ABM_v6()
        amountOfRuns=50
        n_trials=2000
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            #"A_tpeak":"int",
            #"A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,20],
            "b_SLEC":[2,6],
            "ren_E":[0,0.01],
            #"A_tpeak":[0,30],
            #"A_tsigma":[1,10]
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    def ABM_v6_Final_3_optuna(self):
        descr="Final 3 optuna"
        model=Model_ABM_v6()
        amountOfRuns=50
        n_trials=3000
        fixed_params={
            "N_begin":1000,

            #"A_tpeak":14,
            #"A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            "A_tpeak":"int",
            "A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,20],
            "b_SLEC":[2,6],
            "ren_E":[0,0.01],
            "A_tpeak":[0,20],
            "A_tsigma":[1,10]
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    def ABM_v6_Final_4_optuna(self):
        descr="Final 4 optuna"
        model=Model_ABM_v6()
        amountOfRuns=50
        n_trials=2000
        fixed_params={
            "N_begin":1000,

            "A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            #"A_tpeak":"int",
            #"A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,10],
            "b_SLEC":[2,6],
            "ren_E":[0,0.01],
            #"A_tpeak":[0,30],
            #"A_tsigma":[1,10]
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    def ABM_v6_Final_5_optuna(self):
        descr="Final 5 optuna"
        model=Model_ABM_v6()
        amountOfRuns=50
        n_trials=2000
        fixed_params={
            "N_begin":1000,

            #"A_tpeak":14,
            #"A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            "A_tpeak":"int",
            "A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,10],
            "b_SLEC":[2,6],
            "ren_E":[0,0.01],
            "A_tpeak":[1,20],
            "A_tsigma":[1,10]
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")

    def ABM_v6_Final_6_optuna(self):
        descr="Final 6 optuna"
        model=Model_ABM_v6()
        amountOfRuns=50
        n_trials=2000
        fixed_params={
            "N_begin":1000,

            #"A_tpeak":14,
            "A_tsigma":5,
            "A_peak":0.3,

            "b_MPEC":2,
            #"b_SLEC":3,

            #"slec_fraction":0.3,

            "feedback_c":0,

            "d_N":0.0003,
            "d_MPEC":0.02,
            "d_SLEC":0.05,
            "d_S":0.0002,
            "d_C":0.004,
            "d_E":0.01,
            "d_R":0.02,
            "f_S":0.03,
            "f_C":0.05,
            "f_E":0.06,
            "f_R":0.015,
        }
        var_params={
            "slec_fraction":"float",
            "contraction_c":"float",
            "b_SLEC":"int",
            "ren_E":"float",
            "A_tpeak":"int",
            #"A_tsigma":"int"
        }
        var_params_bounds={
            "slec_fraction":[0,1],
            "contraction_c":[1,10],
            "b_SLEC":[2,6],
            "ren_E":[0,0.01],
            "A_tpeak":[1,20],
            #"A_tsigma":[1,10]
        }


        opt_obj=Minimize_optuna(model=model,amountOfRuns=amountOfRuns,fixed_params=fixed_params,var_params=var_params,var_params_bounds=var_params_bounds)

        sol=opt_obj.run(descr=descr,n_trials=n_trials)

        old=pd.read_pickle("Thesis_optimisation/ABM_v6.pkl")
        new=pd.concat([old,sol],ignore_index=True)
        new=new.sort_values(by="loss_s",axis=0,ascending=True).reset_index(drop=True)

        new.to_pickle("Thesis_optimisation/ABM_v6.pkl")





if __name__=="__main__":
    try:
        opt=Thesis_optimisations()

        #opt.ABM_v6_Final_6_optuna()

    except KeyboardInterrupt:
        quit()


