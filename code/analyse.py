
#Bachelor theis: exploratory file: data analysis & visualization
#Author: Tibe Yperman
#E-mail: tibe.yperman@student.uantwerpen.be
#Last revision: 09/05/2026

# NOTE: file cannot be used without the data from the ACTIV study (not yet publicly available)

import pandas as pd
import numpy as np
import os.path
import time
import threading

#Visualization
import dash
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import plotly
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import webbrowser

# analyse.py
import sys

# Add the project root to the path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#Extra files
from exploratory.model_compartments import Model_Compartment
from models_ABM import Model_ABM,Model_ABM_v5,Model_ABM_v5_2,Model_ABM_v6



# Create a web page for visualization
app = Dash(__name__, use_pages=True, pages_folder="", external_stylesheets=[dbc.themes.BOOTSTRAP])

def plotResults():
    """Create a home page that contains links to all the different pages and run the dash application"""
    app.layout = html.Div([
        dash.page_container,
    ])

    homeLayout=html.Div([
        html.Div(
            dcc.Link(f"{page['name']}", href=page["relative_path"])
        ) for page in dash.page_registry.values()
    ])
    dash.register_page("home", path='/', layout=homeLayout)

    # Open the web page in the default browser when the server starts
    def open_browser():
        time.sleep(1)  # wait for server to start
        webbrowser.open("http://127.0.0.1:8050/")
    threading.Thread(target=open_browser).start()
    app.run(debug=True,use_reloader=False)

# NOTE: Analysis of experimental data - does NOT work since data is not uploaded (not yet publicly available)
class Statistics():
    """Class for handling statistical analysis of the data."""
    def __init__(self,folder="Files"):
        """Load the folder containing the data and that will be used to save the statistics."""

        self.TCellTypes=["Tnaive_response","TSCM_response","Tcm_response","Tem_response","Temra_response"]
        self.TCellTypesResponding=["TSCM_response","Tcm_response","Tem_response","Temra_response"]

        # Check if folder exists
        if os.path.exists(folder):
            self.folder=folder
        else:
            raise ValueError("Please enter a valid foldername")
        
        #Load data 
        self._data=pd.read_csv(f"{self.folder}/Data.csv")

        # Check if the statistic file already exists. If not, create a new one containing the mean already
        if os.path.isfile(f"{self.folder}/Stats.csv"):
            self._statistics=pd.read_csv(f"{self.folder}/Stats.csv")
        else:
            df=self.data
            df=df.groupby(by=["Time_Point"])[self.TCellTypes].apply("mean")
            df=df.reset_index()
            df=df.melt(id_vars="Time_Point",value_vars=self.TCellTypes,var_name="TCell_Type",value_name="mean")
            self.statistics=df

        # Check if the file containing the responder analysis already exists. If not, create it.
        if os.path.isfile(f"{self.folder}/Responders.pkl"):
            self._responders=pd.read_pickle(f"{self.folder}/Responders.pkl")
        else:
            self.responders=self.getResponders()
        
    # "data" contains the experimental data
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self,value):
        self._data=value
        value.to_csv(f"{self.folder}/Data.csv",index=False)
    # "statistics" contains the calculated statistics
    @property
    def statistics(self):
        return self._statistics
    @statistics.setter
    def statistics(self,value):
        self._statistics=value
        value.to_csv(f"{self.folder}/Stats.csv",index=False)
    # "responders" contains the responder analysis data
    @property
    def responders(self):
        return self._responders
    @responders.setter
    def responders(self,value):
        self._responders=value
        value.to_pickle(f"{self.folder}/Responders.pkl")


    def giveGeneralInfo(self):
        """Print general information about the data, such as the number of rows, number of vaccinees, number of vaccinees per time point, 
        maximum and minimum values for each T cell type, amount of zero values and check for negative or NaN values."""

        #Get total number of rows
        n_rows=self.data["Time_Point"].count()

        #Get total number of vaccinees
        n_vaccinees=len(self.data["Vaccinee"].unique())

        #Get number of vaccinees per time point
        n_vaccinees_timepoint=self.data.groupby("Time_Point")["Vaccinee"].count()
        n_vaccinees_timepoint=n_vaccinees_timepoint.reset_index()
        n_vaccinees_timepoint=n_vaccinees_timepoint.rename(columns={"Vaccinee":"N_Vaccinees"})

        #Get maximum and minimum value for each TCell type
        max_values=self.data[self.TCellTypes].max().values
        max_indices=self.data[self.TCellTypes].idxmax().values
        max_data=self.data[["Vaccinee","Time_Point"]].loc[max_indices].assign(Max=max_values).reset_index(drop=True)
        max_data.insert(loc=0,column="TCell_Type",value=self.TCellTypes)

        min_values=self.data[self.TCellTypes].min().values
        min_indices=self.data[self.TCellTypes].idxmin().values
        min_data=self.data[["Vaccinee","Time_Point"]].loc[min_indices].assign(Min=min_values).reset_index(drop=True)
        min_data.insert(loc=0,column="TCell_Type",value=self.TCellTypes)

        #Get number of zero values
        zero_values=self.data.groupby("Time_Point")[self.TCellTypes].apply(lambda x: (x==0).sum())
        zero_values_percentage=zero_values[self.TCellTypes].div(other=n_vaccinees_timepoint["N_Vaccinees"].values,axis=0).reset_index()
        zero_values=zero_values.reset_index()

        #Check for negative or nan values
        negative_values=(self.data[self.TCellTypes]<0).any()
        nan_values=self.data[self.TCellTypes].isna().any()

        #Printing results
        print(f"Total number of rows: {n_rows}")
        print(f"Total number of vaccinees: {n_vaccinees}\n")
        print("Number of vaccinees per timepoint:")
        print(n_vaccinees_timepoint, "\n")
        print(f"Maximum values:\n{max_data}\n")
        print(f"Minimum values:\n{min_data}\n")
        print(f"Amount of vaccinees with zero response:\n{zero_values}\n")
        print(f"Percentage of vaccinees with zero response:\n{zero_values_percentage}\n")
        print(f"Check for negative values:\n{negative_values}\n")
        print(f"Check for NaN values:\n{nan_values}")

    def changeTimePoints(self):
        """Add the data of timepoints 15 and 29 to resp. 22 and 43. All measurements at day 356 are added to day 365."""
        self.data=self.data.replace(to_replace={"Time_Point":{15:22,29:43,356:365}})

    def getStatistics(self,statistic="Median"):
        """Function used to calculate certain statistics. Currently only the median is implemented."""
        
        # When the median statistic is called: calculate median, 25th and 75th quantile and IQR if not already calculated
        if statistic=="Median" and not "Median" in self.statistics.columns:
            Median=self.data.groupby(by=["Time_Point"])[self.TCellTypes].quantile(q=0.5).reset_index().melt(id_vars="Time_Point",value_vars=self.TCellTypes,var_name="TCell_Type",value_name="Median")
            self.statistics=self.statistics.merge(Median,how="inner")
            Q25=self.data.groupby(by=["Time_Point"])[self.TCellTypes].quantile(q=0.25).reset_index().melt(id_vars="Time_Point",value_vars=self.TCellTypes,var_name="TCell_Type",value_name="Q25")
            self.statistics=self.statistics.merge(Q25,how="inner")
            Q75=self.data.groupby(by=["Time_Point"])[self.TCellTypes].quantile(q=0.75).reset_index().melt(id_vars="Time_Point",value_vars=self.TCellTypes,var_name="TCell_Type",value_name="Q75")
            self.statistics=self.statistics.merge(Q75,how="inner")
        else:
            print("Statistic already calculated or not recognized.")

    def getResponders(self,TCell_Type="Tem_response"):
        """Responder analysis of a given T cell type."""

        # Check if responder analysis has already been done for the T cell type
        df_already_exists=False
        if hasattr(self,'responders'):
            if TCell_Type in self.responders["TCell_Type"].values:
                return self.responders
            else:
                df_already_exists=True


        df_TCell_Type=self.data[["Vaccinee","Time_Point",TCell_Type]]

        #GET RESPONDERS
        # First, extract each vaccinee's response value at timepoint 1
        baseline = df_TCell_Type[df_TCell_Type["Time_Point"] == 1][["Vaccinee", TCell_Type]]
        baseline = baseline.rename(columns={TCell_Type: "baseline"})
        # Merge baseline values back to full dataset
        df_Compare = df_TCell_Type.merge(baseline, on="Vaccinee")
        # Compare response at each time point with baseline
        df_Compare["is_responder"] = df_Compare[TCell_Type] > (1.1*df_Compare["baseline"])
        # For each vaccinee, check if any timepoint exceeds the baseline
        df_responders = df_Compare.groupby("Vaccinee")["is_responder"].any().reset_index()

        #GET CLASSIC PEAK PATTERN
        # Get vaccinees with measurements at day 22 and 43
        df=df_TCell_Type
        peakers=(
            df.query("Time_Point in [22, 43]")
            .groupby("Vaccinee")[TCell_Type]
            .apply(lambda x: (x > 0).all())
            .pipe(lambda s: s[s].index.tolist())
        )
        # Get vaccinees that have a climb
        df_wide = df[df["Vaccinee"].isin(peakers)].pivot(index="Vaccinee", columns="Time_Point", values=TCell_Type)
        peakers = (df_wide[22] > 1.1 * df_wide[1])
        peakers = peakers[peakers].index.tolist()
        # Get vaccinees that have a drop
        df_wide = df_wide[df_wide.index.isin(peakers)]
        df_peakers = (
            (df_wide[43] < 0.8 * df_wide[22])
            .rename("is_classic_peak")
            .reset_index()
        )
        # Merge into the responders dataframe
        df_responders = df_responders.merge(df_peakers, on="Vaccinee", how="left").fillna(value={"is_classic_peak": False})

        #GET PERSISTANCE
        responders=df_responders[df_responders["is_responder"]==True]["Vaccinee"].unique()
        year_vaccinees=df_TCell_Type[df_TCell_Type["Time_Point"]==365]["Vaccinee"].unique()
        df_wide = df_TCell_Type[df_TCell_Type["Vaccinee"].isin(year_vaccinees)].pivot(index="Vaccinee", columns="Time_Point", values=TCell_Type)
        df_persistors = (
            ((df_wide[365] > 1.1 * df_wide[1])&(df_wide.index.isin(responders)))
            .rename("has_persistence")
            .reset_index()
        )
        df_responders = df_responders.merge(df_persistors, on="Vaccinee", how="left").fillna(value={"has_persistence": "No data"})
        df_responders.insert(0,"TCell_Type",TCell_Type)

        # Add new T cell type to dataframe containing all responder analysis
        if df_already_exists:
            df_responders=pd.concat([self.responders,df_responders],ignore_index=True)
            self.responders=df_responders
        
        return df_responders

    def pathwayBiasAnalysis(self):
        """Analysis of pathway bias: how many cells take effector/memory route."""

        # Analysis is only done on responders, so these are filtered.
        responders=self.responders[self.responders["is_responder"]==True]["Vaccinee"].values
        df=self.data[self.data["Vaccinee"].isin(responders)] #Contains all responders

        #Getting peaks
        df_peaks=df[df["Time_Point"]<60].groupby("Vaccinee")[self.TCellTypes].max().reset_index()

        #Getting ratios
        df_ratios=pd.DataFrame(df_peaks["Vaccinee"])
        df_ratios["early_effector_bias"]=(df_peaks["Tem_response"]+df_peaks["Temra_response"])/(df_peaks["Tem_response"]+df_peaks["Temra_response"]+df_peaks["Tcm_response"]+df_peaks["TSCM_response"]+0.001)
        df_ratios["effector_memory_ratio"]=(df_peaks["Tem_response"]+df_peaks["Temra_response"])/(df_peaks["Tcm_response"]+df_peaks["TSCM_response"]+0.001)

        effector_memory_conditions=[
            df_ratios["effector_memory_ratio"]>2,
            df_ratios["effector_memory_ratio"]<0.5,
            (df_ratios["effector_memory_ratio"]>=0.5)&(df_ratios["effector_memory_ratio"]<=2)
        ]
        effector_memory_choices=["Effector","Memory","Balanced"]
        df_ratios["effector_memory_dominance"]=np.select(effector_memory_conditions,effector_memory_choices,default="Failed")
        #Get 1 year ratio
        df_1year=df[df["Time_Point"]==365].copy() #All responders (with 1 year data) at day 365
        df_1year["memory_persistence"]=(df_1year["Tcm_response"]+df_1year["TSCM_response"])/(df_1year["Tem_response"]+df_1year["Temra_response"]+df_1year["Tcm_response"]+df_1year["TSCM_response"]+0.001)
        
        df_ratios=df_ratios.merge(df_1year[["Vaccinee","memory_persistence"]],how='left',on="Vaccinee").fillna(value={"memory_persistence":"No data"})
        
        return df_ratios
        


    def plotMedian(self):
        """Visualization of the medians."""

        def getFig(df,maxTime,title):
            """General figure function (will be used for 60 days data and 365 days data)"""

            fig = go.Figure()
            # Get a distinct color per TCell_Type using Plotly’s default qualitative palette
            color_map = {
                tcell: color
                for tcell, color in zip(self.TCellTypesResponding,
                                        px.colors.qualitative.Safe)
            }

            # Add each T cell type to the figure
            for tcell in self.TCellTypesResponding:
                sub = df[df["TCell_Type"] == tcell].sort_values("Time_Point")
                color = color_map[tcell]
                # IQR bands
                r, g, b = plotly.colors.unlabel_rgb(color)
                fig.add_trace(go.Scatter(
                    x=pd.concat([sub["Time_Point"], sub["Time_Point"][::-1]]),
                    y=pd.concat([sub["Q25"], sub["Q75"][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=tcell,
                    name=f"{tcell} IQR"
                ))

                # Get response name
                if tcell=="TSCM_response":
                    name="Tscm_response"
                else:
                    name=name
                # Median lines
                fig.add_trace(go.Scatter(
                    x=sub["Time_Point"],
                    y=sub["Median"],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(color=color,size=8),
                    name=name,
                    legendgroup=tcell  # IMPORTANT: links band and line
                ))


            fig.update_layout(
                title=title,
                xaxis_title="Time Point (days)",
                yaxis_title="%AIM\u207A",
                xaxis=dict(range=[0,maxTime]),
                template="plotly_white"
            )

            return fig

        # Get first 60 days figure and full year figure
        fig60days=getFig(df=self.statistics,maxTime=60,title="Median + IQR per T Cell Type: First 60 days")
        fig1year=getFig(df=self.statistics,maxTime=366,title="Median + IQR per T Cell Type: All year")

        # Dash layout
        layout=dbc.Container(
            fluid=True,
            children=[
            dbc.Row([
                dcc.Graph(figure=fig60days,id="ID_plotMedian_graph60")
            ],
            style={"height":"80vh"}
            ),
            dbc.Row([
                dcc.Graph(figure=fig1year,id="ID_plotMedian_graph365")
            ],
            style={"height":"80vh"}
            ),
            dbc.Button(
                "Download figure 60",
                id="ID_plotMedian_download_button60",
                size="sm"
            ),
            dbc.Button(
                "Download figure 365",
                id="ID_plotMedian_download_button365",
                size="sm"
            ),
            dcc.Download("ID_plotMedian_download"),
        ]
        )

        @app.callback(
            Output("ID_plotMedian_download", "data"),

            Input("ID_plotMedian_download_button60", "n_clicks"),

            State("ID_plotMedian_graph60","figure"),
            prevent_initial_call=True,
        )
        def save_figures(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=600),
                "Plot60.png"
            )
        
        @app.callback(
            Output("ID_plotMedian_download", "data",allow_duplicate=True),

            Input("ID_plotMedian_download_button365", "n_clicks"),

            State("ID_plotMedian_graph365","figure"),
            prevent_initial_call=True,
        )
        def save_figures(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=600),
                "Plot365.png"
            )


        dash.register_page("Median", layout=layout)

    def plotSpaghetti(self):
        """Visualization of each seperate T cell type using spaghetti plots."""

        #Get which vaccinees have 1 year data (there will be a button to only show these)
        data=self.data.copy()
        vaccinees_1year=list(data[data["Time_Point"]==365]["Vaccinee"].values)+["Vaccinees"]

        #Get 1year only medium
        data_1year=data[data["Vaccinee"].isin(vaccinees_1year)]
        median_1year=data_1year.groupby(by=["Time_Point"])[self.TCellTypes].quantile(q=0.5).reset_index()

        def getFig(TCell,title):
            """Function to make spaghetti plot of a given T cell type."""
            fig=px.line(data_frame=self.data,x="Time_Point",y=TCell,color="Vaccinee")
            fig.update_layout(
                legend_title="Legend",
                title=title,
                xaxis_title="Time Point (days)",
                yaxis_title="%AIM\u207A",
                xaxis=dict(range=[0,43]),
                template="plotly_white"
            )
            for trace in fig.data:
                trace.line.width=1
                trace.legendgroup = "all_groups"
                if trace.name == "TYF06":
                    trace.name = "Vaccinees"
                    trace.showlegend = True
                else:
                    trace.showlegend = False

            # Add thick median line
            statisticsTCell=self.statistics[self.statistics["TCell_Type"]==TCell]
            fig.add_trace(go.Scatter(
                x=statisticsTCell["Time_Point"],
                y=statisticsTCell["Median"],
                mode="lines",
                line=dict(width=5,color="black"),
                name="Median",
                showlegend=True
            ))

            # Add line that gives medians only calculated of vaccinees with 1 year data
            fig.add_trace(go.Scatter(
                x=median_1year["Time_Point"],
                y=median_1year[TCell],
                mode="lines",
                line=dict(width=5,color="black"),
                name="Median ",
                showlegend=True,
                visible=False
            ))

            return fig

        figTscm=getFig(TCell="TSCM_response",title="Responses of all vaccinees + median of Tscm")
        figTcm=getFig(TCell="Tcm_response",title="Responses of all vaccinees + median of Tcm")
        figTem=getFig(TCell="Tem_response",title="Responses of all vaccinees + median of Tem")
        figTemra=getFig(TCell="Temra_response",title="Responses of all vaccinees + median of Temra")

        # Dash layout
        layout=dbc.Container(
                fluid=True,
                children=[
                    dbc.Row([
                        dbc.Switch(
                            id="ID_spaghetti_1year-toggle",
                            value=False,
                            label="1 year only",
                        )
                    ]),
                    dbc.Row([
                        dcc.Graph(id="ID_spaghetti_Tscm-graph",figure=figTscm)
                    ],
                    style={"height":"70vh"}),
                    dbc.Row([
                        dcc.Graph(id="ID_spaghetti_Tcm-graph",figure=figTcm)
                    ],
                    style={"height":"70vh"}),
                    dbc.Row([
                        dcc.Graph(id="ID_spaghetti_Tem-graph",figure=figTem)
                    ],
                    style={"height":"70vh"}),
                    dbc.Row([
                        dcc.Graph(id="ID_spaghetti_Temra-graph",figure=figTemra)
                    ],
                    style={"height":"70vh"}),
                    dbc.Button(
                        "Download Tscm",
                        id="ID_plotSpaghetti_download_button_Tscm",
                        size="sm"
                    ),
                    dbc.Button(
                        "Download Tcm",
                        id="ID_plotSpaghetti_download_button_Tcm",
                        size="sm"
                    ),
                    dbc.Button(
                        "Download Tem",
                        id="ID_plotSpaghetti_download_button_Tem",
                        size="sm"
                    ),
                    dbc.Button(
                        "Download Temra",
                        id="ID_plotSpaghetti_download_button_Temra",
                        size="sm"
                    ),
                    dcc.Download("ID_plotSpaghetti_download"),
                ])
        
        # Interactive toggle to enable or disable 1 year data
        @callback(
            Output("ID_spaghetti_Tscm-graph","figure"),
            Output("ID_spaghetti_Tcm-graph","figure"),
            Output("ID_spaghetti_Tem-graph","figure"),
            Output("ID_spaghetti_Temra-graph","figure"),

            Input("ID_spaghetti_1year-toggle","value"),
        )
        def updateGraphs(toggle_1year):
            for trace in figTscm.data:
                if not trace.name in vaccinees_1year:
                    trace.visible=not toggle_1year
                if trace.name == "Median ":
                    trace.visible = toggle_1year
            for trace in figTcm.data:
                if not trace.name in vaccinees_1year:
                    trace.visible=not toggle_1year
                if trace.name == "Median ":
                    trace.visible = toggle_1year
            for trace in figTem.data:
                if not trace.name in vaccinees_1year:
                    trace.visible=not toggle_1year
                if trace.name == "Median ":
                    trace.visible = toggle_1year
            for trace in figTemra.data:
                if not trace.name in vaccinees_1year:
                    trace.visible=not toggle_1year
                if trace.name == "Median ":
                    trace.visible = toggle_1year

            if toggle_1year:
                figTscm.update_layout(xaxis=dict(range=[0,365]))
                figTcm.update_layout(xaxis=dict(range=[0,365]))
                figTem.update_layout(xaxis=dict(range=[0,365]))
                figTemra.update_layout(xaxis=dict(range=[0,365]))
            else:
                figTscm.update_layout(xaxis=dict(range=[0,43]))
                figTcm.update_layout(xaxis=dict(range=[0,43]))
                figTem.update_layout(xaxis=dict(range=[0,43]))
                figTemra.update_layout(xaxis=dict(range=[0,43]))

            return figTscm,figTcm,figTem,figTemra

        # DOWNLOAD FIGURES
        @app.callback(
            Output("ID_plotSpaghetti_download", "data"),

            Input("ID_plotSpaghetti_download_button_Tscm", "n_clicks"),

            State("ID_spaghetti_Tscm-graph","figure"),
            prevent_initial_call=True,
        )
        def save_figures(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=600),
                "Tscm_spaghetti.png"
            )
        
        @app.callback(
            Output("ID_plotSpaghetti_download", "data",allow_duplicate=True),

            Input("ID_plotSpaghetti_download_button_Tcm", "n_clicks"),

            State("ID_spaghetti_Tcm-graph","figure"),
            prevent_initial_call=True,
        )
        def save_figures(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=600),
                "Tcm_spaghetti.png"
            )
        
        @app.callback(
            Output("ID_plotSpaghetti_download", "data",allow_duplicate=True),

            Input("ID_plotSpaghetti_download_button_Tem", "n_clicks"),

            State("ID_spaghetti_Tem-graph","figure"),
            prevent_initial_call=True,
        )
        def save_figures(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=600),
                "Tem_spaghetti.png"
            )
        
        @app.callback(
            Output("ID_plotSpaghetti_download", "data",allow_duplicate=True),

            Input("ID_plotSpaghetti_download_button_Temra", "n_clicks"),

            State("ID_spaghetti_Temra-graph","figure"),
            prevent_initial_call=True,
        )
        def save_figures(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=600),
                "Temra_spaghetti.png"
            )

        dash.register_page("Spaghetti", layout=layout)

    def plotResponders_Tem(self):
        """Visualize responder analysis of Tem"""
        df_responders=self.responders[self.responders["TCell_Type"]=="Tem_response"]

        df_resp=df_responders[["Vaccinee","is_responder"]]
        df_other=df_responders[df_responders["is_responder"]==True][["Vaccinee","is_classic_peak","has_persistence"]]

        df_total=df_resp.merge(df_other,how="left")
        
        #Print results
        n_total=len(df_resp)
        n_responders=df_resp["is_responder"].sum()
        n_classic_peak=df_other["is_classic_peak"].sum()
        n_persistors=(df_other["has_persistence"]==True).sum()
        n_no_persistence=(df_other["has_persistence"]==False).sum()

        # print(f"Total number of vaccinees: {n_total}")
        # print(f"Number of responders: {n_responders} ({n_responders/n_total*100:.2f}%)")
        # print(f"Number of classic peak (among responders): {n_classic_peak} ({n_classic_peak/n_responders*100:.2f}%)")
        # print(f"Number of vaccinees with 1 year data: {n_persistors + n_no_persistence}")
        # print(f"Number of persistors (among 1 year data): {n_persistors} ({n_persistors/(n_persistors + n_no_persistence)*100:.2f}%)")

        #Plot
        # Melt the dataframe into long format
        df_long = df_total.melt(
            id_vars="Vaccinee",
            value_vars=["is_responder", "is_classic_peak", "has_persistence"],
            var_name="variable",
            value_name="value"
        )

        # Exclude "No data" only for has_persistence
        df_long = df_long[df_long["value"] != "No data"]
        df_long = df_long.dropna(subset=["value"])

        # Rename things
        variable_names = {
            "is_responder": "Responder",
            "is_classic_peak": "Classic Peak among responders",
            "has_persistence": "Persistence among responders"
        }
        df_long=df_long.replace({"variable": variable_names})

        # Count True/False per variable
        df_counts = df_long.groupby(["variable", "value"],observed=False).size().reset_index(name="count")

        # Create barplot
        fig = px.bar(
            df_counts,
            x="variable",
            y="count",
            color="value",
            barmode="group",
            title="Count of True/False per Response Category",
            category_orders={
                "variable": ["Responder", "Classic Peak among responders", "Persistence among responders"],
                "value": [True, False]       # <-- True bar on the left, False on the right
            },
            labels={"variable": "Category", "count": "Number of Vaccinees", "value": "Legend"},
            template="plotly_white"
        )

        # Dash layout
        layout=dbc.Container([
            dbc.Row([
                html.Pre(
                    f"Total number of vaccinees: {n_total}\n"
                    f"Number of responders: {n_responders} ({n_responders/n_total*100:.2f}%)\n"
                    f"Number of classic peak (among responders): {n_classic_peak} ({n_classic_peak/n_responders*100:.2f}%)\n"
                    f"Number of responders with 1 year data: {n_persistors + n_no_persistence}\n"
                    f"Number of persistors (among responders with 1 year data): {n_persistors} ({n_persistors/(n_persistors + n_no_persistence)*100:.2f}%)"
                )
            ]),
            dbc.Row([
                dcc.Graph(figure=fig)
            ],
            style={"height":"80vh"}) 
        ])
        dash.register_page("Responders Tem", layout=layout)

    def plotResponders(self):
        """Visualization of all T cell types responder analysis"""

        df=self.responders

        #Rename subsets
        df=df.replace({"TCell_Type": {"Tem_response": "Tem", "Temra_response": "Temra","Tcm_response": "Tcm","TSCM_response": "Tscm"}})

        #Get the amount of vaccinees that are responders in all T cell types
        df_combined=df.groupby(["Vaccinee"],observed=False)["is_responder"].sum()
        total_responders_True=(df_combined==4).sum()
        total_responders_False=df_combined.size-total_responders_True

        #Save in dataframe
        total_responders_df=pd.DataFrame([{"TCell_Type":"Total",
                            "is_responder":False,
                            "count":total_responders_False},
                            {"TCell_Type":"Total",
                            "is_responder":True,
                            "count":total_responders_True}])

        #Get seperate T cell type responders and add the total responders
        df_counts=df.groupby(["TCell_Type", "is_responder"],observed=False).size().reset_index(name="count")
        df_counts=pd.concat([df_counts,total_responders_df],ignore_index=True,axis=0)

        # Rename for figure
        df_counts=df_counts.replace({"is_responder":{False:"Non-Responders",True:"Responders"}})

        # Create barplot
        fig = px.bar(
            df_counts,
            x="TCell_Type",
            y="count",
            color="is_responder",
            barmode="group",
            title="Count of responders per T cell type",
            category_orders={
                "TCell_Type": ["Tscm", "Tcm", "Tem", "Temra"],
                "is_responder": ["Responders", "Non-Responders"]       # <-- True bar on the left, False on the right
            },
            labels={"count": "Number of Vaccinees", "TCell_Type":"T cell type"},
            template="plotly_white",
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_layout(
            legend_title_text="Legend"      # <-- rename legend title
        )
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

        # Dash layout
        layout=dbc.Container([
            dbc.Row([
                dcc.Graph(figure=fig,id="ID_plotResponders_figure")
            ],
            style={"height":"70vh"}),
            dbc.Button(
                "Download figure",
                id="ID_plotResponders_download_button",
                size="sm"
            ),
            dcc.Download("ID_plotResponders_download"),
        ])

        @app.callback(
            Output("ID_plotResponders_download", "data",allow_duplicate=True),

            Input("ID_plotResponders_download_button", "n_clicks"),

            State("ID_plotResponders_figure","figure"),
            prevent_initial_call=True,
        )
        def save_figures(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=600),
                "PlotResponders.png"
            )

        dash.register_page("responders", layout=layout,name="Responders")

    def plotPathwayBiasAnalysis(self):
        """Visualization of pathway bias analysis."""

        df=self.pathwayBiasAnalysis()

        # Scatterplot of early effector bias vs effector memory ratio, with log scale on y axis
        fig_early_effector__effector_memory=px.scatter(df,x="early_effector_bias",y="effector_memory_ratio",template="plotly_white")
        fig_early_effector__effector_memory.update_yaxes(
            type='log',
            dtick=1,  # step in powers of 10 (10^0, 10^1, 10^2 ...)
            showline=True
        )    

        # Scatterplot of early effector bias vs memory persistence, with log scale on y axis
        fig_early_effector__memory_persistence=px.scatter(df[df["memory_persistence"]!="No data"],x="early_effector_bias",y="memory_persistence",template="plotly_white")
        
        # Create histogram of effector memory ratio.
        regular_bins=np.arange(0,2.00001,0.25)
        def assign_bin(x):
            if x >= 2:
                return "≥2"
            for i in range(len(regular_bins)-1):
                if regular_bins[i] <= x < regular_bins[i+1]:
                    return f"{regular_bins[i]}-{regular_bins[i+1]}"
            return None
        df['bin'] = df['effector_memory_ratio'].apply(assign_bin)
        bin_order = [f"{regular_bins[i]}-{regular_bins[i+1]}" for i in range(len(regular_bins)-1)] + ["≥2"]
        counts = df['bin'].value_counts().reindex(bin_order, fill_value=0)
        fig_effector_memory_histogram = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            #marker_color=['steelblue']*(len(counts)-1)+['orange']  # color last bin differently
        ))
        fig_effector_memory_histogram.update_layout(
            title="Distribution of effector_memory_ratio",
            xaxis_title="effector_memory_ratio",
            yaxis_title="Amount of responders",
            template="plotly_white"
        )

        # Dash layout
        layout=dbc.Container([
            dbc.Row([
                html.P("Effector memory ratio of vaccinees 12 (=400), 27 (=1150) and 49 (=840) not shown.")
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_early_effector__effector_memory)
                ],width=6),
                dbc.Col([
                    dcc.Graph(figure=fig_early_effector__memory_persistence)
                ],width=6),
            ],style={"height":"80vh","width":"100vw"}),
            dbc.Row([
                dcc.Graph(figure=fig_effector_memory_histogram)
            ],
            style={"height":"80vh","width":"100vw"})
        ],
        style={"margin": "0"},
        )
        dash.register_page("PathwayBiasAnalysis", layout=layout)

# Visualization of the exploratory ODE model
def plotCompareCompartmentModel():
    """Function for visualizing compartment model result compared to data."""

    model_comp=Model_Compartment()
    TCell_Types_Responding=["TSCM_response","Tcm_response","Tem_response","Temra_response"]

    # Dash layout
    layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    # LEFT COLUMN: INPUT BOXES
                    dbc.Col(
                        width=2,
                        children=[
                            dbc.Card(
                                body=True,
                                children=[
                                    html.H6("Parameters"),

                                    dbc.Label("N_begin",size="sm"),
                                    dbc.Input(
                                        id="COCO_N_begin-input",
                                        type="number",
                                        value=1000000,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("A_begin",size="sm"),
                                    dbc.Input(
                                        id="COCO_A_begin-input",
                                        type="number",
                                        value=1,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("A_duration",size="sm"),
                                    dbc.Input(
                                        id="COCO_A_duration-input",
                                        type="number",
                                        value=14,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("p_act",size="sm"),
                                    dbc.Input(
                                        id="COCO_p_act-input",
                                        type="number",
                                        value=0.0001,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("frac_S",size="sm"),
                                    dbc.Input(
                                        id="COCO_frac_S-input",
                                        type="number",
                                        value=0.3,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("p_SC",size="sm"),
                                    dbc.Input(
                                        id="COCO_p_SC-input",
                                        type="number",
                                        value=0.05,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("p_CE",size="sm"),
                                    dbc.Input(
                                        id="COCO_p_CE-input",
                                        type="number",
                                        value=0.05,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("p_ER",size="sm"),
                                    dbc.Input(
                                        id="COCO_p_ER-input",
                                        type="number",
                                        value=0.10,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("r_S",size="sm"),
                                    dbc.Input(
                                        id="COCO_r_S-input",
                                        type="number",
                                        value=0.02,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("r_C",size="sm"),
                                    dbc.Input(
                                        id="COCO_r_C-input",
                                        type="number",
                                        value=0.05,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("r_E",size="sm"),
                                    dbc.Input(
                                        id="COCO_r_E-input",
                                        type="number",
                                        value=0.5,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("d_N",size="sm"),
                                    dbc.Input(
                                        id="COCO_d_N-input",
                                        type="number",
                                        value=0.0005,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("d_S",size="sm"),
                                    dbc.Input(
                                        id="COCO_d_S-input",
                                        type="number",
                                        value=0.001,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("d_C",size="sm"),
                                    dbc.Input(
                                        id="COCO_d_C-input",
                                        type="number",
                                        value=0.002,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("d_E",size="sm"),
                                    dbc.Input(
                                        id="COCO_d_E-input",
                                        type="number",
                                        value=0.3,
                                        debounce=True,
                                        size="sm"
                                    ),

                                    dbc.Label("d_R",size="sm"),
                                    dbc.Input(
                                        id="COCO_d_R-input",
                                        type="number",
                                        value=0.15,
                                        debounce=True,
                                        size="sm"
                                    ),
                                ],
                                style={"height":"80vh", "overflowY": "auto"},
                            ),
                        ],
                    ),

                    # RIGHT COLUMN: GRAPH
                    dbc.Col(
                        width=10,
                        children=[
                            dbc.Card(
                                body=False,
                                children=[
                                    dcc.Graph(id="COCO_line-graph")
                                ],
                                style={"height":"80vh"},
                            ),
                        ],
                    ),
                ],
                className="mt-3",
                #style={"height":"100vh"}
            )
        ],
    )

    # Callback to update graph
    @app.callback(
        Output("COCO_line-graph", "figure"),
        Input("COCO_N_begin-input", "value"),
        Input("COCO_A_begin-input", "value"),
        Input("COCO_A_duration-input", "value"),
        Input("COCO_p_act-input", "value"),
        Input("COCO_frac_S-input", "value"),
        Input("COCO_p_SC-input", "value"),
        Input("COCO_p_CE-input", "value"),
        Input("COCO_p_ER-input", "value"),
        Input("COCO_r_S-input", "value"),
        Input("COCO_r_C-input", "value"),
        Input("COCO_r_E-input", "value"),
        Input("COCO_d_N-input", "value"),
        Input("COCO_d_S-input", "value"),
        Input("COCO_d_C-input", "value"),
        Input("COCO_d_E-input", "value"),
        Input("COCO_d_R-input", "value"),
    )
    def update_graph(N_begin,A_begin,
                 A_duration,
                 p_act,frac_S,
                 p_SC,p_CE,p_ER,
                 r_S, r_C, r_E,
                 d_N, d_S, d_C, d_E, d_R):
        
        # Set model parameters
        model_comp.N_begin=N_begin
        model_comp.A_begin=A_begin
        model_comp.A_duration=A_duration
        model_comp.p_act=p_act
        model_comp.frac_S=frac_S
        model_comp.p_SC=p_SC
        model_comp.p_CE=p_CE
        model_comp.p_ER=p_ER
        model_comp.r_S=r_S
        model_comp.r_C=r_C
        model_comp.r_E=r_E
        model_comp.d_N=d_N
        model_comp.d_S=d_S
        model_comp.d_C=d_C
        model_comp.d_E=d_E
        model_comp.d_R=d_R

        # Run simulation
        data=model_comp.runSimulation()

        # Normalize model (this is not the correct method: we should introduce an analytical scaling factor instead)
        sums=data[TCell_Types_Responding].sum(axis=1)
        max_sum=sums.max()

        data[TCell_Types_Responding]=data[TCell_Types_Responding]/max_sum

        # Drop naive data, as this is not examined in the study
        data=data.drop(columns="Tnaive_response")

        expStats=pd.read_csv("Files/Stats.csv")

        fig=go.Figure()

        color_map = {
                    tcell: color
                    for tcell, color in zip(TCell_Types_Responding,
                                            px.colors.qualitative.Plotly)
                }
        
        # Create figure of model results compared to experimental medians
        for TCell_Type in TCell_Types_Responding:
            data_exp_sub=expStats[expStats["TCell_Type"]==TCell_Type]
            color = color_map[TCell_Type]
            # Get model
            fig.add_trace(go.Scatter(
                x=data["Time_Point"],
                y=data[TCell_Type],
                mode="lines",
                line=dict(color=color),
                name=f"{TCell_Type} model",
                legendgroup=TCell_Type  # links subset model and experiment
            ))
            # Get experiment
            fig.add_trace(go.Scatter(
                x=data_exp_sub["Time_Point"],
                y=data_exp_sub["Median"],
                mode="markers",
                marker=dict(size=10,color=color),
                name=f"{TCell_Type} exp",
                legendgroup=TCell_Type  # links subset model and experiment
            ))

        fig.update_layout(
            template="plotly_white",
            title="Responses: model (lines) and experiment (dots)",
            xaxis_title="Time point (days)",
            yaxis_title="Relative response"
        )


        return fig

    dash.register_page("ComparisonExpCompartmentModel", layout=layout,name="Compartment Model")


# NOTE: 
class CompareModels():
    """Class for visualization of the different models compared to the experimental data"""
    def __init__(self):
        """Initialization of the models and the optimization results."""
        self.TCell_Types_Responding=["TSCM_response","Tcm_response","Tem_response","Temra_response"]
        self.TCell_Types_All=["Tnaive_response","TSCM_response","Tcm_response","Tem_response","Temra_response"]
        
        self.expStats=pd.read_csv("Files/Stats.csv")
        self.data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False).to_numpy() #Same form as data_model
        self.data_exp_normalization=np.broadcast_to(np.max(self.data_exp,0),(4,4)) #Get normalization based on the maximum experimental values
        self.number_of_datapoints=np.size(self.data_exp)

        # Loading optimization results
        self.data_autofit_abm=pd.read_pickle("Thesis optimization/ABM_v1.pkl")
        self.data_autofit_abm_v5=pd.read_pickle("Thesis optimization/ABM_v5.pkl")
        self.data_autofit_abm_v5_2=pd.read_pickle("Thesis optimization/ABM_v5_2.pkl")
        self.data_autofit_abm_v6=pd.read_pickle("Thesis optimization/ABM_v6.pkl")

        # Loading models
        self.model_abm=Model_ABM()
        self.model_abm_v5=Model_ABM_v5()
        self.model_abm_v5_2=Model_ABM_v5_2()
        self.model_abm_v6=Model_ABM_v6()

    # NOTE: the general ABM visualizations were used for exploration and are not fully updated to the latest version of the ABMs, 
    # so these functions may contain mistakes. 
    # The presentation functions were used as a demo in a presentation of the thesis.
    # The autofit files are correct and updated.

    def ABM(self):
        """Interactive visualization of the baseline model"""

        # Dash layout
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),
                                        
                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_s-input",
                                            type="number",
                                            value=0.5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True,
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm"
                                        ),


                                        dbc.Label("A_tpeak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_A_tpeak-input",
                                            type="number",
                                            value=14,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("A_tsigma",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_A_tsigma-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_A_peak-input",
                                            type="number",
                                            value=0.15,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_slec_fraction-input",
                                            type="number",
                                            value=0.6,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_feedback_c-input",
                                            type="number",
                                            value=0.005,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("d_N",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_d_N-input",
                                            type="number",
                                            value=0.0003,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_d_MPEC-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_d_SLEC-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_d_S-input",
                                            type="number",
                                            value=0.0002,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_d_C-input",
                                            type="number",
                                            value=0.004,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_d_E-input",
                                            type="number",
                                            value=0.01,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_d_R-input",
                                            type="number",
                                            value=0.2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_f_S-input",
                                            type="number",
                                            value=0.03,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_f_C-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_f_E-input",
                                            type="number",
                                            value=0.06,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_f_R-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_line-graph")
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_IQR_toggle-input",
                                    value=False,
                                    label="IQR",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_replication_MeanBand_toggle-input",
                                    value=False,
                                    label="Mean Band",
                                )
                            ]
                        ),
                    ]
                )
            ],
        )


        # Run model specified amount of times for given parameters
        @app.callback(
            Output("ID_ABM_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_s-input","value"), 

            Input("ID_ABM_runMultiple-input","n_clicks"),
            State("ID_ABM_amountOfRuns-input","value"),

            State("ID_ABM_N_begin-input", "value"),
            State("ID_ABM_A_tpeak-input", "value"),
            State("ID_ABM_A_tsigma-input", "value"),
            State("ID_ABM_A_peak-input", "value"),
            State("ID_ABM_slec_fraction-input", "value"),
            State("ID_ABM_feedback_c-input", "value"),
            State("ID_ABM_b_MPEC-input", "value"),
            State("ID_ABM_b_SLEC-input", "value"),
            State("ID_ABM_d_N-input", "value"),
            State("ID_ABM_d_MPEC-input", "value"),
            State("ID_ABM_d_SLEC-input", "value"),
            State("ID_ABM_d_S-input", "value"),
            State("ID_ABM_d_C-input", "value"),
            State("ID_ABM_d_E-input", "value"),
            State("ID_ABM_d_R-input", "value"),
            State("ID_ABM_f_S-input", "value"),
            State("ID_ABM_f_C-input", "value"),
            State("ID_ABM_f_E-input", "value"),
            State("ID_ABM_f_R-input", "value"),
            State("ID_ABM_IQR_toggle-input","value"),
            State("ID_ABM_replication_MeanBand_toggle-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_tpeak,A_tsigma,A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            d_N, d_MPEC, d_SLEC, d_S, d_C, d_E, d_R,
            f_S,f_C,f_E,f_R,
            IQR_toggle,MeanBand_toggle
        ):
            # Set model parameters
            self.model_abm.N_begin=N_begin
            self.model_abm.A_tpeak=A_tpeak
            self.model_abm.A_tsigma=A_tsigma
            self.model_abm.A_peak=A_peak
            self.model_abm.slec_fraction=slec_fraction
            self.model_abm.feedback_c=feedback_c
            self.model_abm.b_MPEC=b_MPEC
            self.model_abm.b_SLEC=b_SLEC
            self.model_abm.d_N=d_N
            self.model_abm.d_MPEC=d_MPEC
            self.model_abm.d_SLEC=d_SLEC
            self.model_abm.d_S=d_S
            self.model_abm.d_C=d_C
            self.model_abm.d_E=d_E
            self.model_abm.d_R=d_R
            self.model_abm.f_S=f_S
            self.model_abm.f_C=f_C
            self.model_abm.f_E=f_E
            self.model_abm.f_R=f_R

            # Simulate model multiple times
            data=self.model_abm.simulateMultiple_MP(days=365,amount=amountOfRuns)

            # Get median data
            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()
        
            # Calculate global scaling factor
            s=calculate_s(dataMean)

            dataMean=dataMean*s
            dataStd=data.groupby(["Time_Point"])[self.TCell_Types_Responding].std()*s
            dataStdUp=dataMean+dataStd
            dataStdDown=dataMean-dataStd
            dataTimePoints=dataMean.index

            return getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,IQR_toggle,MeanBand_toggle),s

        def getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,
                                IQR_toggle,MeanBand_toggle):
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_All,
                                                px.colors.qualitative.Plotly)
                    }
            
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                # --- IQR band ---
                r, g, b = plotly.colors.hex_to_rgb(color)
                fig.add_trace(go.Scatter(
                    x=pd.concat([data_exp_sub["Time_Point"], data_exp_sub["Time_Point"][::-1]]),
                    y=pd.concat([data_exp_sub["Q25"], data_exp_sub["Q75"][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdUp[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdDown[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response"
            )


            return fig

        def calculate_s(data):
            #Calculate numerator
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
            numerator_s=product_s.sum()

            #Calculate denominator
            square_s=data_model**2
            denominator_s=square_s.sum()

            #Calculate s
            s=numerator_s/denominator_s

            return s

        #Changing visuals
        @app.callback(
            Output("ID_ABM_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_IQR_toggle-input","value"),
            Input("ID_ABM_replication_MeanBand_toggle-input","value"),

            State("ID_ABM_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(IQR_toggle,MeanBand_toggle,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="IQR":
                    if IQR_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="MeanBand":
                    if MeanBand_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

            return fig



        dash.register_page("ABMModel", layout=layout,name="ABM")
    
    def ABM_v5(self):
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),
                                        
                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_s-input",
                                            type="number",
                                            value=0.5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm"
                                        ),


                                        dbc.Label("A_tpeak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_A_tpeak-input",
                                            type="number",
                                            value=14,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("A_tsigma",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_A_tsigma-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_A_peak-input",
                                            type="number",
                                            value=0.15,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_slec_fraction-input",
                                            type="number",
                                            value=0.6,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_feedback_c-input",
                                            type="number",
                                            value=0.00005,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("d_N",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_d_N-input",
                                            type="number",
                                            value=0.0003,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_d_MPEC-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_d_SLEC-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_d_S-input",
                                            type="number",
                                            value=0.0002,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_d_C-input",
                                            type="number",
                                            value=0.004,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_d_E-input",
                                            type="number",
                                            value=0.01,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_d_R-input",
                                            type="number",
                                            value=0.2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_f_S-input",
                                            type="number",
                                            value=0.03,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_f_C-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_f_E-input",
                                            type="number",
                                            value=0.06,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_f_R-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_v5_line-graph")
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_v5_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_v5_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v5_IQR_toggle-input",
                                    value=False,
                                    label="IQR",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v5_replication_MeanBand_toggle-input",
                                    value=False,
                                    label="Mean Band",
                                )
                            ]
                        ),
                    ]
                )
            ],
        )


        #REPLICATION
        @app.callback(
            Output("ID_ABM_v5_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_v5_s-input","value"), 

            Input("ID_ABM_v5_runMultiple-input","n_clicks"),
            State("ID_ABM_v5_amountOfRuns-input","value"),

            State("ID_ABM_v5_N_begin-input", "value"),
            State("ID_ABM_v5_A_tpeak-input", "value"),
            State("ID_ABM_v5_A_tsigma-input", "value"),
            State("ID_ABM_v5_A_peak-input", "value"),
            State("ID_ABM_v5_slec_fraction-input", "value"),
            State("ID_ABM_v5_feedback_c-input", "value"),
            State("ID_ABM_v5_b_MPEC-input", "value"),
            State("ID_ABM_v5_b_SLEC-input", "value"),
            State("ID_ABM_v5_d_N-input", "value"),
            State("ID_ABM_v5_d_MPEC-input", "value"),
            State("ID_ABM_v5_d_SLEC-input", "value"),
            State("ID_ABM_v5_d_S-input", "value"),
            State("ID_ABM_v5_d_C-input", "value"),
            State("ID_ABM_v5_d_E-input", "value"),
            State("ID_ABM_v5_d_R-input", "value"),
            State("ID_ABM_v5_f_S-input", "value"),
            State("ID_ABM_v5_f_C-input", "value"),
            State("ID_ABM_v5_f_E-input", "value"),
            State("ID_ABM_v5_f_R-input", "value"),
            State("ID_ABM_v5_IQR_toggle-input","value"),
            State("ID_ABM_v5_replication_MeanBand_toggle-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_tpeak,A_tsigma,A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            d_N, d_MPEC, d_SLEC, d_S, d_C, d_E, d_R,
            f_S,f_C,f_E,f_R,
            IQR_toggle,MeanBand_toggle
        ):
            self.model_abm_v5.N_begin=N_begin
            self.model_abm_v5.A_tpeak=A_tpeak
            self.model_abm_v5.A_tsigma=A_tsigma
            self.model_abm_v5.A_peak=A_peak
            self.model_abm_v5.slec_fraction=slec_fraction
            self.model_abm_v5.feedback_c=feedback_c
            self.model_abm_v5.b_MPEC=b_MPEC
            self.model_abm_v5.b_SLEC=b_SLEC
            self.model_abm_v5.d_N=d_N
            self.model_abm_v5.d_MPEC=d_MPEC
            self.model_abm_v5.d_SLEC=d_SLEC
            self.model_abm_v5.d_S=d_S
            self.model_abm_v5.d_C=d_C
            self.model_abm_v5.d_E=d_E
            self.model_abm_v5.d_R=d_R
            self.model_abm_v5.f_S=f_S
            self.model_abm_v5.f_C=f_C
            self.model_abm_v5.f_E=f_E
            self.model_abm_v5.f_R=f_R


            data=self.model_abm_v5.simulateMultiple_MP(days=365,amount=amountOfRuns)

            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()
        
            s,loss_s=calculate_s(dataMean)

            dataMean=dataMean*s
            dataStd=data.groupby(["Time_Point"])[self.TCell_Types_Responding].std()*s
            dataStdUp=dataMean+dataStd
            dataStdDown=dataMean-dataStd
            dataTimePoints=dataMean.index

            return getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,IQR_toggle,MeanBand_toggle),s

        def getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,
                                IQR_toggle,MeanBand_toggle):
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_All,
                                                px.colors.qualitative.Plotly)
                    }
            
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                # --- IQR band ---
                r, g, b = plotly.colors.hex_to_rgb(color)
                fig.add_trace(go.Scatter(
                    x=pd.concat([data_exp_sub["Time_Point"], data_exp_sub["Time_Point"][::-1]]),
                    y=pd.concat([data_exp_sub["Q25"], data_exp_sub["Q75"][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdUp[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdDown[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response"
            )


            return fig

        def calculate_s(data):
            #Calculate numerator
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
            numerator_s=product_s.sum()

            #Calculate denominator
            square_s=data_model**2
            denominator_s=square_s.sum()

            #Calculate s
            s=numerator_s/(denominator_s+0.00001)

            #Calculate loss
            loss_matrix=(data_exp-s*data_model)**2
            loss_s=loss_matrix.sum()

            return s,loss_s

        #Changing visuals
        @app.callback(
            Output("ID_ABM_v5_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_v5_IQR_toggle-input","value"),
            Input("ID_ABM_v5_replication_MeanBand_toggle-input","value"),

            State("ID_ABM_v5_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(IQR_toggle,MeanBand_toggle,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="IQR":
                    if IQR_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="MeanBand":
                    if MeanBand_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

            return fig



        dash.register_page("ABMModel-v5", layout=layout,name="ABM v5")

    def ABM_v6(self):
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),
                                        
                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_s-input",
                                            type="number",
                                            value=0.5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm"
                                        ),


                                        dbc.Label("A_tpeak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_A_tpeak-input",
                                            type="number",
                                            value=14,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("A_tsigma",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_A_tsigma-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_A_peak-input",
                                            type="number",
                                            value=0.15,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_slec_fraction-input",
                                            type="number",
                                            value=0.6,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_feedback_c-input",
                                            type="number",
                                            value=0.00005,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("contraction_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_contraction_c-input",
                                            type="number",
                                            value=10,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("ren_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_ren_E-input",
                                            type="number",
                                            value=0.005,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("d_N",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_d_N-input",
                                            type="number",
                                            value=0.0003,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_d_MPEC-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_d_SLEC-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_d_S-input",
                                            type="number",
                                            value=0.0002,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_d_C-input",
                                            type="number",
                                            value=0.004,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_d_E-input",
                                            type="number",
                                            value=0.01,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_d_R-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_f_S-input",
                                            type="number",
                                            value=0.03,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_f_C-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_f_E-input",
                                            type="number",
                                            value=0.06,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_f_R-input",
                                            type="number",
                                            value=0.015,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_v6_line-graph")
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_v6_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_v6_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v6_IQR_toggle-input",
                                    value=False,
                                    label="IQR",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v6_replication_MeanBand_toggle-input",
                                    value=False,
                                    label="Mean Band",
                                )
                            ]
                        ),
                    ]
                )
            ],
        )


        #REPLICATION
        @app.callback(
            Output("ID_ABM_v6_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_v6_s-input","value"), 

            Input("ID_ABM_v6_runMultiple-input","n_clicks"),
            State("ID_ABM_v6_amountOfRuns-input","value"),

            State("ID_ABM_v6_N_begin-input", "value"),
            State("ID_ABM_v6_A_tpeak-input", "value"),
            State("ID_ABM_v6_A_tsigma-input", "value"),
            State("ID_ABM_v6_A_peak-input", "value"),
            State("ID_ABM_v6_slec_fraction-input", "value"),
            State("ID_ABM_v6_feedback_c-input", "value"),
            State("ID_ABM_v6_b_MPEC-input", "value"),
            State("ID_ABM_v6_b_SLEC-input", "value"),
            State("ID_ABM_v6_d_N-input", "value"),
            State("ID_ABM_v6_d_MPEC-input", "value"),
            State("ID_ABM_v6_d_SLEC-input", "value"),
            State("ID_ABM_v6_d_S-input", "value"),
            State("ID_ABM_v6_d_C-input", "value"),
            State("ID_ABM_v6_d_E-input", "value"),
            State("ID_ABM_v6_d_R-input", "value"),
            State("ID_ABM_v6_f_S-input", "value"),
            State("ID_ABM_v6_f_C-input", "value"),
            State("ID_ABM_v6_f_E-input", "value"),
            State("ID_ABM_v6_f_R-input", "value"),
            State("ID_ABM_v6_contraction_c-input", "value"),
            State("ID_ABM_v6_ren_E-input", "value"),

            State("ID_ABM_v6_IQR_toggle-input","value"),
            State("ID_ABM_v6_replication_MeanBand_toggle-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_tpeak,A_tsigma,A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            d_N, d_MPEC, d_SLEC, d_S, d_C, d_E, d_R,
            f_S,f_C,f_E,f_R,
            contraction_c,
            ren_E,
            IQR_toggle,MeanBand_toggle
        ):
            self.model_abm_v6.N_begin=N_begin
            self.model_abm_v6.A_tpeak=A_tpeak
            self.model_abm_v6.A_tsigma=A_tsigma
            self.model_abm_v6.A_peak=A_peak
            self.model_abm_v6.slec_fraction=slec_fraction
            self.model_abm_v6.feedback_c=feedback_c
            self.model_abm_v6.b_MPEC=b_MPEC
            self.model_abm_v6.b_SLEC=b_SLEC
            self.model_abm_v6.d_N=d_N
            self.model_abm_v6.d_MPEC=d_MPEC
            self.model_abm_v6.d_SLEC=d_SLEC
            self.model_abm_v6.d_S=d_S
            self.model_abm_v6.d_C=d_C
            self.model_abm_v6.d_E=d_E
            self.model_abm_v6.d_R=d_R
            self.model_abm_v6.f_S=f_S
            self.model_abm_v6.f_C=f_C
            self.model_abm_v6.f_E=f_E
            self.model_abm_v6.f_R=f_R
            self.model_abm_v6.contraction_c=contraction_c
            self.model_abm_v6.ren_E=ren_E


            data=self.model_abm_v6.simulateMultiple_MP(days=365,amount=amountOfRuns)

            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()
        
            s,loss_s=calculate_s(dataMean)

            dataMean=dataMean*s
            dataStd=data.groupby(["Time_Point"])[self.TCell_Types_Responding].std()*s
            dataStdUp=dataMean+dataStd
            dataStdDown=dataMean-dataStd
            dataTimePoints=dataMean.index

            return getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,IQR_toggle,MeanBand_toggle),s

        def getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,
                                IQR_toggle,MeanBand_toggle):
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_All,
                                                px.colors.qualitative.Plotly)
                    }
            
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                # --- IQR band ---
                r, g, b = plotly.colors.hex_to_rgb(color)
                fig.add_trace(go.Scatter(
                    x=pd.concat([data_exp_sub["Time_Point"], data_exp_sub["Time_Point"][::-1]]),
                    y=pd.concat([data_exp_sub["Q25"], data_exp_sub["Q75"][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdUp[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdDown[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response"
            )


            return fig

        def calculate_s(data):
            #Calculate numerator
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
            numerator_s=product_s.sum()

            #Calculate denominator
            square_s=data_model**2
            denominator_s=square_s.sum()

            #Calculate s
            s=numerator_s/(denominator_s+0.00001)

            #Calculate loss
            loss_matrix=(data_exp-s*data_model)**2
            loss_s=loss_matrix.sum()

            return s,loss_s

        #Changing visuals
        @app.callback(
            Output("ID_ABM_v6_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_v6_IQR_toggle-input","value"),
            Input("ID_ABM_v6_replication_MeanBand_toggle-input","value"),

            State("ID_ABM_v6_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(IQR_toggle,MeanBand_toggle,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="IQR":
                    if IQR_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="MeanBand":
                    if MeanBand_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

            return fig



        dash.register_page("ABMModel-v6", layout=layout,name="ABM v6")


    def ABM_presentation(self):
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),
                                        
                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_presentation_s-input",
                                            type="number",
                                            value=0.5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_presentation_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm"
                                        ),


                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_presentation_A_peak-input",
                                            type="number",
                                            value=0.15,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_presentation_feedback_c-input",
                                            type="number",
                                            value=0.005,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_presentation_slec_fraction-input",
                                            type="number",
                                            value=0.6,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_presentation_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_presentation_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_presentation_line-graph")
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_presentation_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_presentation_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_presentation_exp_toggle-input",
                                    value=False,
                                    label="Experimental data",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_presentation_IQR_toggle-input",
                                    value=False,
                                    label="IQR",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_presentation_replication_MeanBand_toggle-input",
                                    value=False,
                                    label="Mean Band",
                                )
                            ]
                        ),
                    ]
                )
            ],
        )


        #REPLICATION
        @app.callback(
            Output("ID_ABM_presentation_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_presentation_s-input","value"), 

            Input("ID_ABM_presentation_runMultiple-input","n_clicks"),
            State("ID_ABM_presentation_amountOfRuns-input","value"),

            State("ID_ABM_presentation_N_begin-input", "value"),
            State("ID_ABM_presentation_A_peak-input", "value"),
            State("ID_ABM_presentation_slec_fraction-input", "value"),
            State("ID_ABM_presentation_feedback_c-input", "value"),
            State("ID_ABM_presentation_b_MPEC-input", "value"),
            State("ID_ABM_presentation_b_SLEC-input", "value"),

            State("ID_ABM_presentation_exp_toggle-input","value"),
            State("ID_ABM_presentation_IQR_toggle-input","value"),
            State("ID_ABM_presentation_replication_MeanBand_toggle-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            exp_toggle,IQR_toggle,MeanBand_toggle
        ):
            self.model_abm.N_begin=N_begin
            self.model_abm.A_peak=A_peak
            self.model_abm.slec_fraction=slec_fraction
            self.model_abm.feedback_c=feedback_c
            self.model_abm.b_MPEC=b_MPEC
            self.model_abm.b_SLEC=b_SLEC


            data=self.model_abm.simulateMultiple_MP(days=365,amount=amountOfRuns)

            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_All].median()
        
            s,loss_s=calculate_s(dataMean)

            dataMean=dataMean*s
            dataStd=data.groupby(["Time_Point"])[self.TCell_Types_All].std()*s
            dataStdUp=dataMean+dataStd
            dataStdDown=dataMean-dataStd
            dataTimePoints=dataMean.index

            return getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,exp_toggle,IQR_toggle,MeanBand_toggle),s

        def getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,
                                exp_toggle,IQR_toggle,MeanBand_toggle):
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_All,
                                                px.colors.qualitative.Plotly)
                    }
            
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    visible=exp_toggle
                ))
                # --- IQR band ---
                r, g, b = plotly.colors.hex_to_rgb(color)
                fig.add_trace(go.Scatter(
                    x=pd.concat([data_exp_sub["Time_Point"], data_exp_sub["Time_Point"][::-1]]),
                    y=pd.concat([data_exp_sub["Q25"], data_exp_sub["Q75"][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdUp[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdDown[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response",
                xaxis_range=[-2,367]
            )


            return fig

        def calculate_s(data):
            #Calculate numerator
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats.pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
            numerator_s=product_s.sum()

            #Calculate denominator
            square_s=data_model**2
            denominator_s=square_s.sum()

            #Calculate s
            s=numerator_s/denominator_s

            #Calculate loss
            loss_matrix=(data_exp-s*data_model)**2
            loss_s=loss_matrix.sum()

            return s,loss_s

        #Changing visuals
        @app.callback(
            Output("ID_ABM_presentation_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_presentation_exp_toggle-input","value"),
            Input("ID_ABM_presentation_IQR_toggle-input","value"),
            Input("ID_ABM_presentation_replication_MeanBand_toggle-input","value"),

            State("ID_ABM_presentation_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(exp_toggle,IQR_toggle,MeanBand_toggle,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="exp":
                    if exp_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="IQR":
                    if IQR_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="MeanBand":
                    if MeanBand_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

            return fig



        dash.register_page("ABMModelPresentation", layout=layout,name="ABM presentation")

    def ABM_v6_presentation(self):
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),
                                        
                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_s-input",
                                            type="number",
                                            value=0.5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm"
                                        ),


                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_A_peak-input",
                                            type="number",
                                            value=0.3,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_feedback_c-input",
                                            type="number",
                                            value=0.001,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_slec_fraction-input",
                                            type="number",
                                            value=0.2,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("contraction_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_contraction_c-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm"
                                        ),

                                        dbc.Label("ren_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_ren_E-input",
                                            type="number",
                                            value=0.001,
                                            debounce=True,
                                            size="sm"
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_v6_presentation_line-graph")
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_v6_presentation_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_v6_presentation_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v6_presentation_exp_toggle-input",
                                    value=False,
                                    label="Experimental data",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v6_presentation_IQR_toggle-input",
                                    value=False,
                                    label="IQR",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v6_presentation_replication_MeanBand_toggle-input",
                                    value=False,
                                    label="Mean Band",
                                )
                            ]
                        ),
                    ]
                )
            ],
        )


        #REPLICATION
        @app.callback(
            Output("ID_ABM_v6_presentation_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_v6_presentation_s-input","value"), 

            Input("ID_ABM_v6_presentation_runMultiple-input","n_clicks"),
            State("ID_ABM_v6_presentation_amountOfRuns-input","value"),

            State("ID_ABM_v6_presentation_N_begin-input", "value"),
            State("ID_ABM_v6_presentation_A_peak-input", "value"),
            State("ID_ABM_v6_presentation_slec_fraction-input", "value"),
            State("ID_ABM_v6_presentation_feedback_c-input", "value"),
            State("ID_ABM_v6_presentation_b_MPEC-input", "value"),
            State("ID_ABM_v6_presentation_b_SLEC-input", "value"),
            State("ID_ABM_v6_presentation_contraction_c-input", "value"),
            State("ID_ABM_v6_presentation_ren_E-input", "value"),

            State("ID_ABM_v6_presentation_exp_toggle-input","value"),
            State("ID_ABM_v6_presentation_IQR_toggle-input","value"),
            State("ID_ABM_v6_presentation_replication_MeanBand_toggle-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            contraction_c,
            ren_E,
            exp_toggle,IQR_toggle,MeanBand_toggle
        ):
            self.model_abm_v6.N_begin=N_begin
            self.model_abm_v6.A_peak=A_peak
            self.model_abm_v6.slec_fraction=slec_fraction
            self.model_abm_v6.feedback_c=feedback_c
            self.model_abm_v6.b_MPEC=b_MPEC
            self.model_abm_v6.b_SLEC=b_SLEC
            self.model_abm_v6.contraction_c=contraction_c
            self.model_abm_v6.ren_E=ren_E


            data=self.model_abm_v6.simulateMultiple_MP(days=365,amount=amountOfRuns)

            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()
        
            s=calculate_s(dataMean)

            dataMean=dataMean*s
            dataStd=data.groupby(["Time_Point"])[self.TCell_Types_Responding].std()*s
            dataStdUp=dataMean+dataStd
            dataStdDown=dataMean-dataStd
            dataTimePoints=dataMean.index

            return getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,exp_toggle,IQR_toggle,MeanBand_toggle),s

        def getReplicationGraph(dataTimePoints,dataMean,dataStdUp,dataStdDown,
                                exp_toggle,IQR_toggle,MeanBand_toggle):
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_All,
                                                px.colors.qualitative.Plotly)
                    }
            
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    visible=exp_toggle
                ))
                # --- IQR band ---
                r, g, b = plotly.colors.hex_to_rgb(color)
                fig.add_trace(go.Scatter(
                    x=pd.concat([data_exp_sub["Time_Point"], data_exp_sub["Time_Point"][::-1]]),
                    y=pd.concat([data_exp_sub["Q25"], data_exp_sub["Q75"][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdUp[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataStdDown[TCell_Type],
                    mode="lines",
                    line=dict(color=color,dash="dash"),
                    name=f"{TCell_Type} MeanBand",
                    legendgroup=TCell_Type,  # links subset model and experiment
                    showlegend=False,
                    hoverinfo="skip",
                    visible=MeanBand_toggle,
                ))
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response",
                xaxis_range=[-2,367]
            )


            return fig

        def calculate_s(data):
            #Calculate numerator
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
            numerator_s=product_s.sum()

            #Calculate denominator
            square_s=data_model**2
            denominator_s=square_s.sum()

            #Calculate s
            s=numerator_s/denominator_s

            return s

        #Changing visuals
        @app.callback(
            Output("ID_ABM_v6_presentation_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_v6_presentation_exp_toggle-input","value"),
            Input("ID_ABM_v6_presentation_IQR_toggle-input","value"),
            Input("ID_ABM_v6_presentation_replication_MeanBand_toggle-input","value"),

            State("ID_ABM_v6_presentation_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(exp_toggle,IQR_toggle,MeanBand_toggle,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="exp":
                    if exp_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="IQR":
                    if IQR_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="MeanBand":
                    if MeanBand_toggle==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

            return fig



        dash.register_page("ABMModel-v6-Presentation", layout=layout,name="ABM v6 presentation")



    def ABM_autofit(self):
        """Visualization of the baseline model results."""

        # Dash layout
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),
                                        
                                        dbc.Label("RMSE",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_RMSE-input",
                                            type="number",
                                            value=np.nan,
                                            debounce=True,
                                            size="sm",
                                            disabled=True,
                                        ),

                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_s-input",
                                            type="number",
                                            value=np.nan,
                                            debounce=True,
                                            size="sm",
                                            disabled=True,
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),


                                        dbc.Label("A_tpeak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_A_tpeak-input",
                                            type="number",
                                            value=14,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("A_tsigma",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_A_tsigma-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_A_peak-input",
                                            type="number",
                                            value=0.15,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_slec_fraction-input",
                                            type="number",
                                            value=0.6,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_feedback_c-input",
                                            type="number",
                                            value=0.005,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_N",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_d_N-input",
                                            type="number",
                                            value=0.0003,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_d_MPEC-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_d_SLEC-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_d_S-input",
                                            type="number",
                                            value=0.0002,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_d_C-input",
                                            type="number",
                                            value=0.004,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_d_E-input",
                                            type="number",
                                            value=0.01,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_d_R-input",
                                            type="number",
                                            value=0.2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_f_S-input",
                                            type="number",
                                            value=0.03,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_f_C-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_f_E-input",
                                            type="number",
                                            value=0.06,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_autofit_f_R-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_autofit_line-graph",
                                                  figure=go.Figure(),
                                                  config={"responsive": True},
                                                  style={"height": "100%", "width": "100%"},
                                                  )
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Button(
                                            "Load solution",
                                            id="ID_ABM_autofit_Load",
                                            n_clicks=0
                                        ),
                                        dbc.Modal([
                                            dbc.ModalHeader("Select a row"),
                                            dbc.ModalBody(
                                                dash_table.DataTable(
                                                    id="ID_ABM_autofit_Table",
                                                    data=self.data_autofit_abm[["Description","loss_s","s"]].to_dict("records"),
                                                    columns=[{"name":col,"id":col} for col in self.data_autofit_abm[["Description","loss_s","s"]].columns],
                                                    row_selectable="single",
                                                    style_table={"overflowX": "auto","overflowY":"auto"}
                                                )
                                            )],
                                            id="ID_ABM_autofit_Table_Modal",
                                            is_open=False,
                                            size="lg"
                                        )
                                    ],
                                )
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_autofit_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_autofit_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_autofit_IQR_exp-input",
                                    value=False,
                                    label="IQR exp",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_autofit_IQR_model-input",
                                    value=False,
                                    label="IQR model",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Button(
                                    "Download figure",
                                    id="ID_ABM_autofit_download_figure-button",
                                    size="sm"
                                ),
                                dcc.Download("ID_ABM_autofit_download"),
                            ]
                        ),
                    ]
                )
            ],
        )


        # Plot model results for given parameters
        @app.callback(
            Output("ID_ABM_autofit_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_autofit_s-input","value"), 
            Output("ID_ABM_autofit_RMSE-input","value"), 

            Input("ID_ABM_autofit_runMultiple-input","n_clicks"),
            State("ID_ABM_autofit_amountOfRuns-input","value"),

            State("ID_ABM_autofit_N_begin-input", "value"),
            State("ID_ABM_autofit_A_tpeak-input", "value"),
            State("ID_ABM_autofit_A_tsigma-input", "value"),
            State("ID_ABM_autofit_A_peak-input", "value"),
            State("ID_ABM_autofit_slec_fraction-input", "value"),
            State("ID_ABM_autofit_feedback_c-input", "value"),
            State("ID_ABM_autofit_b_MPEC-input", "value"),
            State("ID_ABM_autofit_b_SLEC-input", "value"),
            State("ID_ABM_autofit_d_N-input", "value"),
            State("ID_ABM_autofit_d_MPEC-input", "value"),
            State("ID_ABM_autofit_d_SLEC-input", "value"),
            State("ID_ABM_autofit_d_S-input", "value"),
            State("ID_ABM_autofit_d_C-input", "value"),
            State("ID_ABM_autofit_d_E-input", "value"),
            State("ID_ABM_autofit_d_R-input", "value"),
            State("ID_ABM_autofit_f_S-input", "value"),
            State("ID_ABM_autofit_f_C-input", "value"),
            State("ID_ABM_autofit_f_E-input", "value"),
            State("ID_ABM_autofit_f_R-input", "value"),
            State("ID_ABM_autofit_IQR_exp-input","value"),
            State("ID_ABM_autofit_IQR_model-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_tpeak,A_tsigma,A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            d_N, d_MPEC, d_SLEC, d_S, d_C, d_E, d_R,
            f_S,f_C,f_E,f_R,
            IQR_exp,IQR_model
        ):
            # Set model parameters
            self.model_abm.N_begin=N_begin
            self.model_abm.A_tpeak=A_tpeak
            self.model_abm.A_tsigma=A_tsigma
            self.model_abm.A_peak=A_peak
            self.model_abm.slec_fraction=slec_fraction
            self.model_abm.feedback_c=feedback_c
            self.model_abm.b_MPEC=b_MPEC
            self.model_abm.b_SLEC=b_SLEC
            self.model_abm.d_N=d_N
            self.model_abm.d_MPEC=d_MPEC
            self.model_abm.d_SLEC=d_SLEC
            self.model_abm.d_S=d_S
            self.model_abm.d_C=d_C
            self.model_abm.d_E=d_E
            self.model_abm.d_R=d_R
            self.model_abm.f_S=f_S
            self.model_abm.f_C=f_C
            self.model_abm.f_E=f_E
            self.model_abm.f_R=f_R

            # Get model results
            data=self.model_abm.simulateMultiple_MP(days=365,amount=amountOfRuns)

            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()

            # Get scaling factor and normalized RMSE
            s,RMSE=calculate_s(dataMean)

            # Get IQR
            dataMean=dataMean*s
            dataQ25=data.groupby(["Time_Point"])[self.TCell_Types_Responding].quantile(q=0.25)*s
            dataQ75=data.groupby(["Time_Point"])[self.TCell_Types_Responding].quantile(q=0.75)*s

            dataTimePoints=list(dataMean.index)

            #Get antigen
            antigen_time=np.array(dataTimePoints)
            antigen_values=dataMean[self.TCell_Types_Responding].max(axis=None)*np.exp(-0.5*((antigen_time-A_tpeak)/A_tsigma)**2)

            return getReplicationGraph(dataTimePoints,dataMean,dataQ25,dataQ75,IQR_exp,IQR_model,antigen_values),s,RMSE

        def getReplicationGraph(dataTimePoints,dataMean,dataQ25,dataQ75,
                                IQR_exp,IQR_model,
                                antigen_values):
            """Get the graph showing the model results"""
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_Responding,
                                                px.colors.qualitative.Safe)
                    }
            # Add model + experiment for each T cell type
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                #Experimental data
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type,
                      
                    # IQR exp
                    error_y=dict(
                        type="data",
                        array=data_exp_sub["Q75"]-data_exp_sub["Median"],
                        arrayminus=data_exp_sub["Median"]-data_exp_sub["Q25"],

                        visible=IQR_exp
                    )
                ))
                # IQR model
                r, g, b = plotly.colors.unlabel_rgb(color)
                fig.add_trace(go.Scatter(
                    x=list(dataTimePoints)+list(dataTimePoints[::-1]),
                    y=list(dataQ25[TCell_Type].values)+list(dataQ75[TCell_Type].values[::-1]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_model,
                ))

            #Add antigen curve --- the plotted absolute values are not correct (it's just to show the shape of the curve)
            fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=antigen_values,
                    mode="lines",
                    line=dict(color="blue"),
                    #line=dict(color=color,dash="dash"),
                    name=f"Antigen",
                    #legendgroup=TCell_Type,  # links subset model and experiment
                    #showlegend=True,
                    #hoverinfo="skip",
                    visible="legendonly",
                ))
            
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response",
                autosize=True
            )


            return fig

        def calculate_s(data):
            """Calculate the scaling factor s and the normalized RMSE for the model results compared to the experimental data."""

            #Calculate numerator of scaling factor formula
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
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

            return s,RMSE

        #Changing visuals: enabling/disabling IQRs
        @app.callback(
            Output("ID_ABM_autofit_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_autofit_IQR_exp-input","value"),
            Input("ID_ABM_autofit_IQR_model-input","value"),

            State("ID_ABM_autofit_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(IQR_exp,IQR_model,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if len(name)==1:
                    continue

                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="IQR":
                    if IQR_model==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="exp":
                    trace.error_y.visible=IQR_exp

            return fig



        #Importing a parameter solution found by the optimizer
        # Open the table containing the optimal solutions
        @app.callback(
            Output("ID_ABM_autofit_Table_Modal","is_open"),
            Input("ID_ABM_autofit_Load","n_clicks"),
            State("ID_ABM_autofit_Table_Modal","is_open"),

            prevent_initial_call=True
        )
        def open_modal_table(n,is_open):
            return not is_open
        
        # Selecting a solution from the table -> import the parameters
        @app.callback(
            Output("ID_ABM_autofit_Table_Modal","is_open",allow_duplicate=True),

            Output("ID_ABM_autofit_N_begin-input", "value"),
            Output("ID_ABM_autofit_A_tpeak-input", "value"),
            Output("ID_ABM_autofit_A_tsigma-input", "value"),
            Output("ID_ABM_autofit_A_peak-input", "value"),
            Output("ID_ABM_autofit_slec_fraction-input", "value"),
            Output("ID_ABM_autofit_feedback_c-input", "value"),
            Output("ID_ABM_autofit_b_MPEC-input", "value"),
            Output("ID_ABM_autofit_b_SLEC-input", "value"),
            Output("ID_ABM_autofit_d_N-input", "value"),
            Output("ID_ABM_autofit_d_MPEC-input", "value"),
            Output("ID_ABM_autofit_d_SLEC-input", "value"),
            Output("ID_ABM_autofit_d_S-input", "value"),
            Output("ID_ABM_autofit_d_C-input", "value"),
            Output("ID_ABM_autofit_d_E-input", "value"),
            Output("ID_ABM_autofit_d_R-input", "value"),
            Output("ID_ABM_autofit_f_S-input", "value"),
            Output("ID_ABM_autofit_f_C-input", "value"),
            Output("ID_ABM_autofit_f_E-input", "value"),
            Output("ID_ABM_autofit_f_R-input", "value"),

            Input("ID_ABM_autofit_Table","selected_rows"),

            prevent_initial_call=True
        )
        def load_row(selected_rows):
            row=selected_rows[0]

            params=self.data_autofit_abm["Parameters"][row].copy()
            #Parameter order
            order=["N_begin","A_tpeak","A_tsigma","A_peak","slec_fraction","feedback_c","b_MPEC","b_SLEC",
                    "d_N","d_MPEC","d_SLEC","d_S","d_C","d_E","d_R","f_S","f_C","f_E","f_R"]
            params_list=[]
            for key in order:
                params_list.append(params[key])


            return tuple([False]+params_list)
            

        # Save the figure
        @app.callback(
            Output("ID_ABM_autofit_download", "data"),

            Input("ID_ABM_autofit_download_figure-button", "n_clicks"),

            State("ID_ABM_autofit_line-graph","figure"),
            prevent_initial_call=True,
        )
        def save_plot(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=800),
                "Plot.png"
            )


        dash.register_page("ABMautofit", layout=layout,name="ABM autofit")

    def ABM_v5_autofit(self):
        """Visualization of the FDD model"""

        # Dash layout
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),
                                        
                                        dbc.Label("RMSE",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_RMSE-input",
                                            type="number",
                                            value=np.nan,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_s-input",
                                            type="number",
                                            value=np.nan,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),


                                        dbc.Label("A_tpeak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_A_tpeak-input",
                                            type="number",
                                            value=14,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("A_tsigma",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_A_tsigma-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_A_peak-input",
                                            type="number",
                                            value=0.15,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_slec_fraction-input",
                                            type="number",
                                            value=0.6,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_feedback_c-input",
                                            type="number",
                                            value=0.005,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_N",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_d_N-input",
                                            type="number",
                                            value=0.0003,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_d_MPEC-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_d_SLEC-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_d_S-input",
                                            type="number",
                                            value=0.0002,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_d_C-input",
                                            type="number",
                                            value=0.004,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_d_E-input",
                                            type="number",
                                            value=0.01,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_d_R-input",
                                            type="number",
                                            value=0.2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_f_S-input",
                                            type="number",
                                            value=0.03,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_f_C-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_f_E-input",
                                            type="number",
                                            value=0.06,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_f_R-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_v5_autofit_line-graph",
                                                  figure=go.Figure(),
                                                  config={"responsive": True},
                                                  style={"height": "100%", "width": "100%"},
                                                  )
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Button(
                                            "Load solution",
                                            id="ID_ABM_v5_autofit_Load",
                                            n_clicks=0
                                        ),
                                        dbc.Modal([
                                            dbc.ModalHeader("Select a row"),
                                            dbc.ModalBody(
                                                dash_table.DataTable(
                                                    id="ID_ABM_v5_autofit_Table",
                                                    data=self.data_autofit_abm_v5[["Description","loss_s","s"]].to_dict("records"),
                                                    columns=[{"name":col,"id":col} for col in self.data_autofit_abm_v5[["Description","loss_s","s"]].columns],
                                                    row_selectable="single",
                                                    style_table={"overflowX": "auto","overflowY":"auto"}
                                                )
                                            )],
                                            id="ID_ABM_v5_autofit_Table_Modal",
                                            is_open=False,
                                            size="lg"
                                        )
                                    ],
                                )
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_v5_autofit_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_v5_autofit_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v5_autofit_IQR_exp-input",
                                    value=False,
                                    label="IQR exp",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v5_autofit_IQR_model-input",
                                    value=False,
                                    label="IQR model",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Button(
                                    "Download figure",
                                    id="ID_ABM_v5_autofit_download_figure-button",
                                    size="sm"
                                ),
                                dcc.Download("ID_ABM_v5_autofit_download"),
                            ]
                        ),
                    ]
                )
            ],
        )


        # Plot model results for given parameters
        @app.callback(
            Output("ID_ABM_v5_autofit_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_v5_autofit_s-input","value"), 
            Output("ID_ABM_v5_autofit_RMSE-input","value"), 

            Input("ID_ABM_v5_autofit_runMultiple-input","n_clicks"),
            State("ID_ABM_v5_autofit_amountOfRuns-input","value"),

            State("ID_ABM_v5_autofit_N_begin-input", "value"),
            State("ID_ABM_v5_autofit_A_tpeak-input", "value"),
            State("ID_ABM_v5_autofit_A_tsigma-input", "value"),
            State("ID_ABM_v5_autofit_A_peak-input", "value"),
            State("ID_ABM_v5_autofit_slec_fraction-input", "value"),
            State("ID_ABM_v5_autofit_feedback_c-input", "value"),
            State("ID_ABM_v5_autofit_b_MPEC-input", "value"),
            State("ID_ABM_v5_autofit_b_SLEC-input", "value"),
            State("ID_ABM_v5_autofit_d_N-input", "value"),
            State("ID_ABM_v5_autofit_d_MPEC-input", "value"),
            State("ID_ABM_v5_autofit_d_SLEC-input", "value"),
            State("ID_ABM_v5_autofit_d_S-input", "value"),
            State("ID_ABM_v5_autofit_d_C-input", "value"),
            State("ID_ABM_v5_autofit_d_E-input", "value"),
            State("ID_ABM_v5_autofit_d_R-input", "value"),
            State("ID_ABM_v5_autofit_f_S-input", "value"),
            State("ID_ABM_v5_autofit_f_C-input", "value"),
            State("ID_ABM_v5_autofit_f_E-input", "value"),
            State("ID_ABM_v5_autofit_f_R-input", "value"),
            State("ID_ABM_v5_autofit_IQR_exp-input","value"),
            State("ID_ABM_v5_autofit_IQR_model-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_tpeak,A_tsigma,A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            d_N, d_MPEC, d_SLEC, d_S, d_C, d_E, d_R,
            f_S,f_C,f_E,f_R,
            IQR_exp,IQR_model
        ):
            # Set model parameters
            self.model_abm_v5.N_begin=N_begin
            self.model_abm_v5.A_tpeak=A_tpeak
            self.model_abm_v5.A_tsigma=A_tsigma
            self.model_abm_v5.A_peak=A_peak
            self.model_abm_v5.slec_fraction=slec_fraction
            self.model_abm_v5.feedback_c=feedback_c
            self.model_abm_v5.b_MPEC=b_MPEC
            self.model_abm_v5.b_SLEC=b_SLEC
            self.model_abm_v5.d_N=d_N
            self.model_abm_v5.d_MPEC=d_MPEC
            self.model_abm_v5.d_SLEC=d_SLEC
            self.model_abm_v5.d_S=d_S
            self.model_abm_v5.d_C=d_C
            self.model_abm_v5.d_E=d_E
            self.model_abm_v5.d_R=d_R
            self.model_abm_v5.f_S=f_S
            self.model_abm_v5.f_C=f_C
            self.model_abm_v5.f_E=f_E
            self.model_abm_v5.f_R=f_R

            # Get model results 
            data=self.model_abm_v5.simulateMultiple_MP(days=365,amount=amountOfRuns)

            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()

            # Get scaling factor and normalizedRMSE
            s,RMSE=calculate_s(dataMean)

            dataMean=dataMean*s
            # Get IQR
            dataQ25=data.groupby(["Time_Point"])[self.TCell_Types_Responding].quantile(q=0.25)*s
            dataQ75=data.groupby(["Time_Point"])[self.TCell_Types_Responding].quantile(q=0.75)*s

            dataTimePoints=list(dataMean.index)

            #Get antigen
            antigen_time=np.array(dataTimePoints)
            antigen_values=dataMean[self.TCell_Types_Responding].max(axis=None)*np.exp(-0.5*((antigen_time-A_tpeak)/A_tsigma)**2)

            return getReplicationGraph(dataTimePoints,dataMean,dataQ25,dataQ75,IQR_exp,IQR_model,antigen_values),s,RMSE

        # Get the graph showing the model results
        def getReplicationGraph(dataTimePoints,dataMean,dataQ25,dataQ75,
                                IQR_exp,IQR_model,
                                antigen_values):
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_Responding,
                                                px.colors.qualitative.Safe)
                    }
            
            # Add model + experiment for each T cell type
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                #Experimental data
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type,
                      
                    # IQR exp
                    error_y=dict(
                        type="data",
                        array=data_exp_sub["Q75"]-data_exp_sub["Median"],
                        arrayminus=data_exp_sub["Median"]-data_exp_sub["Q25"],

                        visible=IQR_exp
                    )
                ))
                # IQR model
                r, g, b = plotly.colors.unlabel_rgb(color)
                fig.add_trace(go.Scatter(
                    x=list(dataTimePoints)+list(dataTimePoints[::-1]),
                    y=list(dataQ25[TCell_Type].values)+list(dataQ75[TCell_Type].values[::-1]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_model,
                ))

            #Add antigen curve --- the plotted absolute values are not correct (it's just to show the shape of the curve)
            fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=antigen_values,
                    mode="lines",
                    line=dict(color="blue"),
                    #line=dict(color=color,dash="dash"),
                    name=f"Antigen",
                    #legendgroup=TCell_Type,  # links subset model and experiment
                    #showlegend=True,
                    #hoverinfo="skip",
                    visible="legendonly",
                ))
            
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response",
                autosize=True
            )


            return fig

        def calculate_s(data):
            """Calculate the scaling factor s and RMSE for the model results compared to the experimental data."""

            #Calculate numerator of scaling factor formula
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
            numerator_s=product_s.sum()

            #Calculate denominator
            square_s=data_model**2
            denominator_s=square_s.sum()

            #Calculate s
            s=numerator_s/denominator_s

            #Calculate loss and normalized RMSE
            loss_matrix=((self.data_exp-s*data_model)/self.data_exp_normalization)**2
            SSE=loss_matrix.sum()
            RMSE=np.sqrt(SSE/self.number_of_datapoints)

            return s,RMSE

        #Changing visuals: enabling/disabling IQRs
        @app.callback(
            Output("ID_ABM_v5_autofit_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_v5_autofit_IQR_exp-input","value"),
            Input("ID_ABM_v5_autofit_IQR_model-input","value"),

            State("ID_ABM_v5_autofit_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(IQR_exp,IQR_model,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if len(name)==1:
                    continue

                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="IQR":
                    if IQR_model==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="exp":
                    trace.error_y.visible=IQR_exp

            return fig



        # Importing a parameter solution found by the optimizer
        # Open the table containing the optimal solutions
        @app.callback(
            Output("ID_ABM_v5_autofit_Table_Modal","is_open"),
            Input("ID_ABM_v5_autofit_Load","n_clicks"),
            State("ID_ABM_v5_autofit_Table_Modal","is_open"),

            prevent_initial_call=True
        )
        def open_modal_table(n,is_open):
            return not is_open
        
        # Selecting a solution from the table -> import the parameters
        @app.callback(
            Output("ID_ABM_v5_autofit_Table_Modal","is_open",allow_duplicate=True),

            Output("ID_ABM_v5_autofit_N_begin-input", "value"),
            Output("ID_ABM_v5_autofit_A_tpeak-input", "value"),
            Output("ID_ABM_v5_autofit_A_tsigma-input", "value"),
            Output("ID_ABM_v5_autofit_A_peak-input", "value"),
            Output("ID_ABM_v5_autofit_slec_fraction-input", "value"),
            Output("ID_ABM_v5_autofit_feedback_c-input", "value"),
            Output("ID_ABM_v5_autofit_b_MPEC-input", "value"),
            Output("ID_ABM_v5_autofit_b_SLEC-input", "value"),
            Output("ID_ABM_v5_autofit_d_N-input", "value"),
            Output("ID_ABM_v5_autofit_d_MPEC-input", "value"),
            Output("ID_ABM_v5_autofit_d_SLEC-input", "value"),
            Output("ID_ABM_v5_autofit_d_S-input", "value"),
            Output("ID_ABM_v5_autofit_d_C-input", "value"),
            Output("ID_ABM_v5_autofit_d_E-input", "value"),
            Output("ID_ABM_v5_autofit_d_R-input", "value"),
            Output("ID_ABM_v5_autofit_f_S-input", "value"),
            Output("ID_ABM_v5_autofit_f_C-input", "value"),
            Output("ID_ABM_v5_autofit_f_E-input", "value"),
            Output("ID_ABM_v5_autofit_f_R-input", "value"),

            Input("ID_ABM_v5_autofit_Table","selected_rows"),

            prevent_initial_call=True
        )
        def load_row(selected_rows):
            row=selected_rows[0]

            params=self.data_autofit_abm_v5["Parameters"][row].copy()
            #Parameter order
            order=["N_begin","A_tpeak","A_tsigma","A_peak","slec_fraction","feedback_c","b_MPEC","b_SLEC",
                    "d_N","d_MPEC","d_SLEC","d_S","d_C","d_E","d_R","f_S","f_C","f_E","f_R"]
            params_list=[]
            for key in order:
                params_list.append(params[key])


            return tuple([False]+params_list)
            
        # Download the figure
        @app.callback(
            Output("ID_ABM_v5_autofit_download", "data"),

            Input("ID_ABM_v5_autofit_download_figure-button", "n_clicks"),

            State("ID_ABM_v5_autofit_line-graph","figure"),
            prevent_initial_call=True,
        )
        def save_plot(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=800),
                "Plot.png"
            )


        dash.register_page("ABM-v5-autofit", layout=layout,name="ABM v5 autofit")

    def ABM_v5_2_autofit(self):
        """Visualization of the FDD+Tem renewal model"""

        # Dash layout
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),

                                        dbc.Label("RMSE",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_RMSE-input",
                                            type="number",
                                            value=np.nan,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_s-input",
                                            type="number",
                                            value=np.nan,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),


                                        dbc.Label("A_tpeak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_A_tpeak-input",
                                            type="number",
                                            value=14,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("A_tsigma",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_A_tsigma-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_A_peak-input",
                                            type="number",
                                            value=0.15,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_slec_fraction-input",
                                            type="number",
                                            value=0.6,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_feedback_c-input",
                                            type="number",
                                            value=0.005,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("ren_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_ren_E-input",
                                            type="number",
                                            value=0.005,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_N",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_d_N-input",
                                            type="number",
                                            value=0.0003,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_d_MPEC-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_d_SLEC-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_d_S-input",
                                            type="number",
                                            value=0.0002,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_d_C-input",
                                            type="number",
                                            value=0.004,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_d_E-input",
                                            type="number",
                                            value=0.01,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_d_R-input",
                                            type="number",
                                            value=0.2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_f_S-input",
                                            type="number",
                                            value=0.03,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_f_C-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_f_E-input",
                                            type="number",
                                            value=0.06,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_f_R-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_v5_2_autofit_line-graph",
                                                  figure=go.Figure(),
                                                  config={"responsive": True},
                                                  style={"height": "100%", "width": "100%"},
                                                  )
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Button(
                                            "Load solution",
                                            id="ID_ABM_v5_2_autofit_Load",
                                            n_clicks=0
                                        ),
                                        dbc.Modal([
                                            dbc.ModalHeader("Select a row"),
                                            dbc.ModalBody(
                                                dash_table.DataTable(
                                                    id="ID_ABM_v5_2_autofit_Table",
                                                    data=self.data_autofit_abm_v5_2[["Description","loss_s","s"]].to_dict("records"),
                                                    columns=[{"name":col,"id":col} for col in self.data_autofit_abm_v5_2[["Description","loss_s","s"]].columns],
                                                    row_selectable="single",
                                                    style_table={"overflowX": "auto","overflowY":"auto"}
                                                )
                                            )],
                                            id="ID_ABM_v5_2_autofit_Table_Modal",
                                            is_open=False,
                                            size="lg"
                                        )
                                    ],
                                )
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_v5_2_autofit_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_v5_2_autofit_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v5_2_autofit_IQR_exp-input",
                                    value=False,
                                    label="IQR exp",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v5_2_autofit_IQR_model-input",
                                    value=False,
                                    label="IQR model",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Button(
                                    "Download figure",
                                    id="ID_ABM_v5_2_autofit_download_figure-button",
                                    size="sm"
                                ),
                                dcc.Download("ID_ABM_v5_2_autofit_download"),
                            ]
                        ),
                    ]
                )
            ],
        )


        # Plot model results for given parameters
        @app.callback(
            Output("ID_ABM_v5_2_autofit_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_v5_2_autofit_s-input","value"), 
            Output("ID_ABM_v5_2_autofit_RMSE-input","value"), 

            Input("ID_ABM_v5_2_autofit_runMultiple-input","n_clicks"),
            State("ID_ABM_v5_2_autofit_amountOfRuns-input","value"),

            State("ID_ABM_v5_2_autofit_N_begin-input", "value"),
            State("ID_ABM_v5_2_autofit_A_tpeak-input", "value"),
            State("ID_ABM_v5_2_autofit_A_tsigma-input", "value"),
            State("ID_ABM_v5_2_autofit_A_peak-input", "value"),
            State("ID_ABM_v5_2_autofit_slec_fraction-input", "value"),
            State("ID_ABM_v5_2_autofit_feedback_c-input", "value"),
            State("ID_ABM_v5_2_autofit_b_MPEC-input", "value"),
            State("ID_ABM_v5_2_autofit_b_SLEC-input", "value"),
            State("ID_ABM_v5_2_autofit_ren_E-input", "value"),
            State("ID_ABM_v5_2_autofit_d_N-input", "value"),
            State("ID_ABM_v5_2_autofit_d_MPEC-input", "value"),
            State("ID_ABM_v5_2_autofit_d_SLEC-input", "value"),
            State("ID_ABM_v5_2_autofit_d_S-input", "value"),
            State("ID_ABM_v5_2_autofit_d_C-input", "value"),
            State("ID_ABM_v5_2_autofit_d_E-input", "value"),
            State("ID_ABM_v5_2_autofit_d_R-input", "value"),
            State("ID_ABM_v5_2_autofit_f_S-input", "value"),
            State("ID_ABM_v5_2_autofit_f_C-input", "value"),
            State("ID_ABM_v5_2_autofit_f_E-input", "value"),
            State("ID_ABM_v5_2_autofit_f_R-input", "value"),
            State("ID_ABM_v5_2_autofit_IQR_exp-input","value"),
            State("ID_ABM_v5_2_autofit_IQR_model-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_tpeak,A_tsigma,A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            ren_E,
            d_N, d_MPEC, d_SLEC, d_S, d_C, d_E, d_R,
            f_S,f_C,f_E,f_R,
            IQR_exp,IQR_model
        ):
            # Set model parameters
            self.model_abm_v5_2.N_begin=N_begin
            self.model_abm_v5_2.A_tpeak=A_tpeak
            self.model_abm_v5_2.A_tsigma=A_tsigma
            self.model_abm_v5_2.A_peak=A_peak
            self.model_abm_v5_2.slec_fraction=slec_fraction
            self.model_abm_v5_2.feedback_c=feedback_c
            self.model_abm_v5_2.b_MPEC=b_MPEC
            self.model_abm_v5_2.b_SLEC=b_SLEC
            self.model_abm_v5_2.ren_E=ren_E
            self.model_abm_v5_2.d_N=d_N
            self.model_abm_v5_2.d_MPEC=d_MPEC
            self.model_abm_v5_2.d_SLEC=d_SLEC
            self.model_abm_v5_2.d_S=d_S
            self.model_abm_v5_2.d_C=d_C
            self.model_abm_v5_2.d_E=d_E
            self.model_abm_v5_2.d_R=d_R
            self.model_abm_v5_2.f_S=f_S
            self.model_abm_v5_2.f_C=f_C
            self.model_abm_v5_2.f_E=f_E
            self.model_abm_v5_2.f_R=f_R

            # Get model results
            data=self.model_abm_v5_2.simulateMultiple_MP(days=365,amount=amountOfRuns)

            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()
        
            # Get scaling factor and normalized RMSE
            s,RMSE=calculate_s(dataMean)

            dataMean=dataMean*s
            # Get IQR
            dataQ25=data.groupby(["Time_Point"])[self.TCell_Types_Responding].quantile(q=0.25)*s
            dataQ75=data.groupby(["Time_Point"])[self.TCell_Types_Responding].quantile(q=0.75)*s

            dataTimePoints=list(dataMean.index)

            #Get antigen
            antigen_time=np.array(dataTimePoints)
            antigen_values=dataMean[self.TCell_Types_Responding].max(axis=None)*np.exp(-0.5*((antigen_time-A_tpeak)/A_tsigma)**2)

            return getReplicationGraph(dataTimePoints,dataMean,dataQ25,dataQ75,IQR_exp,IQR_model,antigen_values),s,RMSE

        def getReplicationGraph(dataTimePoints,dataMean,dataQ25,dataQ75,
                                IQR_exp,IQR_model,
                                antigen_values):
            """Get the graph showing the model results and experimental data."""
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_Responding,
                                                px.colors.qualitative.Safe)
                    }
            
            # Add model + experiment for each T cell type
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                #Experimental data
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type,
                      
                    # IQR exp
                    error_y=dict(
                        type="data",
                        array=data_exp_sub["Q75"]-data_exp_sub["Median"],
                        arrayminus=data_exp_sub["Median"]-data_exp_sub["Q25"],

                        visible=IQR_exp
                    )
                ))
                # IQR model
                r, g, b = plotly.colors.unlabel_rgb(color)
                fig.add_trace(go.Scatter(
                    x=list(dataTimePoints)+list(dataTimePoints[::-1]),
                    y=list(dataQ25[TCell_Type].values)+list(dataQ75[TCell_Type].values[::-1]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_model,
                ))

            #Add antigen curve --- the plotted absolute values are not correct (it's just to show the shape of the curve)
            fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=antigen_values,
                    mode="lines",
                    line=dict(color="blue"),
                    #line=dict(color=color,dash="dash"),
                    name=f"Antigen",
                    #legendgroup=TCell_Type,  # links subset model and experiment
                    #showlegend=True,
                    #hoverinfo="skip",
                    visible="legendonly",
                ))
            
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response",
                autosize=True
            )


            return fig

        def calculate_s(data):
            """Calculate the scaling factor s and RMSE for the model results compared to the experimental data."""

            #Calculate numerator of scaling factor formula
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
            numerator_s=product_s.sum()

            #Calculate denominator
            square_s=data_model**2
            denominator_s=square_s.sum()

            #Calculate s
            s=numerator_s/denominator_s

            #Calculate loss and normalized RMSE
            loss_matrix=((self.data_exp-s*data_model)/self.data_exp_normalization)**2
            SSE=loss_matrix.sum()
            RMSE=np.sqrt(SSE/self.number_of_datapoints)

            return s,RMSE

        # Changing visuals: enabling/disabling IQRs
        @app.callback(
            Output("ID_ABM_v5_2_autofit_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_v5_2_autofit_IQR_exp-input","value"),
            Input("ID_ABM_v5_2_autofit_IQR_model-input","value"),

            State("ID_ABM_v5_2_autofit_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(IQR_exp,IQR_model,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if len(name)==1:
                    continue

                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="IQR":
                    if IQR_model==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="exp":
                    trace.error_y.visible=IQR_exp

            return fig



        # Importing a parameter solution found by the optimizer
        # Open the table containing the optimal solutions
        @app.callback(
            Output("ID_ABM_v5_2_autofit_Table_Modal","is_open"),
            Input("ID_ABM_v5_2_autofit_Load","n_clicks"),
            State("ID_ABM_v5_2_autofit_Table_Modal","is_open"),

            prevent_initial_call=True
        )
        def open_modal_table(n,is_open):
            return not is_open
        
        # Selecting a solution from the table -> import the parameters
        @app.callback(
            Output("ID_ABM_v5_2_autofit_Table_Modal","is_open",allow_duplicate=True),

            Output("ID_ABM_v5_2_autofit_N_begin-input", "value"),
            Output("ID_ABM_v5_2_autofit_A_tpeak-input", "value"),
            Output("ID_ABM_v5_2_autofit_A_tsigma-input", "value"),
            Output("ID_ABM_v5_2_autofit_A_peak-input", "value"),
            Output("ID_ABM_v5_2_autofit_slec_fraction-input", "value"),
            Output("ID_ABM_v5_2_autofit_feedback_c-input", "value"),
            Output("ID_ABM_v5_2_autofit_b_MPEC-input", "value"),
            Output("ID_ABM_v5_2_autofit_b_SLEC-input", "value"),
            Output("ID_ABM_v5_2_autofit_ren_E-input", "value"),
            Output("ID_ABM_v5_2_autofit_d_N-input", "value"),
            Output("ID_ABM_v5_2_autofit_d_MPEC-input", "value"),
            Output("ID_ABM_v5_2_autofit_d_SLEC-input", "value"),
            Output("ID_ABM_v5_2_autofit_d_S-input", "value"),
            Output("ID_ABM_v5_2_autofit_d_C-input", "value"),
            Output("ID_ABM_v5_2_autofit_d_E-input", "value"),
            Output("ID_ABM_v5_2_autofit_d_R-input", "value"),
            Output("ID_ABM_v5_2_autofit_f_S-input", "value"),
            Output("ID_ABM_v5_2_autofit_f_C-input", "value"),
            Output("ID_ABM_v5_2_autofit_f_E-input", "value"),
            Output("ID_ABM_v5_2_autofit_f_R-input", "value"),

            Input("ID_ABM_v5_2_autofit_Table","selected_rows"),

            prevent_initial_call=True
        )
        def load_row(selected_rows):
            row=selected_rows[0]

            params=self.data_autofit_abm_v5_2["Parameters"][row].copy()
            #Parameter order
            order=["N_begin","A_tpeak","A_tsigma","A_peak","slec_fraction","feedback_c","b_MPEC","b_SLEC","ren_E",
                    "d_N","d_MPEC","d_SLEC","d_S","d_C","d_E","d_R","f_S","f_C","f_E","f_R"]
            params_list=[]
            for key in order:
                params_list.append(params[key])


            return tuple([False]+params_list)
            
        # Download the figure
        @app.callback(
            Output("ID_ABM_v5_2_autofit_download", "data"),

            Input("ID_ABM_v5_2_autofit_download_figure-button", "n_clicks"),

            State("ID_ABM_v5_2_autofit_line-graph","figure"),
            prevent_initial_call=True,
        )
        def save_plot(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=800),
                "Plot.png"
            )

        dash.register_page("ABM-v5_2-autofit", layout=layout,name="ABM v5_2 autofit")


    def ABM_v6_autofit(self):
        """Visualization of the ADDC model"""

        # Dash layout
        layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        # LEFT COLUMN: INPUT BOXES
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        html.H6("Parameters"),

                                        dbc.Label("RMSE",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_RMSE-input",
                                            type="number",
                                            value=np.nan,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),
                                        
                                        dbc.Label("s (AIM conversion)",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_s-input",
                                            type="number",
                                            value=np.nan,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("N_begin",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_N_begin-input",
                                            type="number",
                                            value=1000,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),


                                        dbc.Label("A_tpeak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_A_tpeak-input",
                                            type="number",
                                            value=14,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("A_tsigma",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_A_tsigma-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("A_peak",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_A_peak-input",
                                            type="number",
                                            value=0.15,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("slec_fraction",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_slec_fraction-input",
                                            type="number",
                                            value=0.6,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("feedback_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_feedback_c-input",
                                            type="number",
                                            value=0.00005,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("b_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_b_MPEC-input",
                                            type="number",
                                            value=2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("b_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_b_SLEC-input",
                                            type="number",
                                            value=5,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("contraction_c",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_contraction_c-input",
                                            type="number",
                                            value=10,
                                            debounce=True,
                                            size="sm",
                                            disabled=False
                                        ),

                                        dbc.Label("ren_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_ren_E-input",
                                            type="number",
                                            value=0.005,
                                            debounce=True,
                                            size="sm",
                                            disabled=False
                                        ),

                                        dbc.Label("d_N",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_d_N-input",
                                            type="number",
                                            value=0.0003,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_MPEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_d_MPEC-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_SLEC",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_d_SLEC-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_d_S-input",
                                            type="number",
                                            value=0.0002,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_d_C-input",
                                            type="number",
                                            value=0.004,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_d_E-input",
                                            type="number",
                                            value=0.01,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("d_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_d_R-input",
                                            type="number",
                                            value=0.2,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_S",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_f_S-input",
                                            type="number",
                                            value=0.03,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_C",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_f_C-input",
                                            type="number",
                                            value=0.05,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),

                                        dbc.Label("f_E",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_f_E-input",
                                            type="number",
                                            value=0.06,
                                            debounce=True,
                                            size="sm",
                                            disabled=False
                                        ),

                                        dbc.Label("f_R",size="sm"),
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_f_R-input",
                                            type="number",
                                            value=0.02,
                                            debounce=True,
                                            size="sm",
                                            disabled=True
                                        ),
                                    ],
                                    style={"height":"80vh", "overflowY": "auto"},
                                ),
                            ],
                        ),

                        # RIGHT COLUMN: GRAPH
                        dbc.Col(
                            width=10,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dcc.Graph(id="ID_ABM_v6_autofit_line-graph",
                                                  figure=go.Figure(),
                                                  config={"responsive": True},
                                                  style={"height": "100%", "width": "100%"},
                                                  )
                                    ],
                                    style={"height":"80vh"},
                                ),
                            ],
                        ),
                    ],
                    className="my-2"
                    #className="mt-3",
                    #style={"height":"100vh"}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Button(
                                            "Load solution",
                                            id="ID_ABM_v6_autofit_Load",
                                            n_clicks=0
                                        ),
                                        dbc.Modal([
                                            dbc.ModalHeader("Select a row"),
                                            dbc.ModalBody(
                                                dash_table.DataTable(
                                                    id="ID_ABM_v6_autofit_Table",
                                                    data=self.data_autofit_abm_v6[["Description","loss_s","s"]].to_dict("records"),
                                                    columns=[{"name":col,"id":col} for col in self.data_autofit_abm_v6[["Description","loss_s","s"]].columns],
                                                    row_selectable="single",
                                                    style_table={"overflowX": "auto","overflowY":"auto"}
                                                )
                                            )],
                                            id="ID_ABM_v6_autofit_Table_Modal",
                                            is_open=False,
                                            size="xl"
                                        )
                                    ],
                                )
                            ]
                        ),
                        #REPLICATION
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[
                                        dbc.Input(
                                            id="ID_ABM_v6_autofit_amountOfRuns-input",
                                            type="number",
                                            placeholder="Amount of runs",
                                            debounce=True,
                                            size="sm",
                                            className="me-5"
                                        ),
                                        dbc.Button(
                                            "Run",
                                            id="ID_ABM_v6_autofit_runMultiple-input",
                                            size="sm"
                                        ),
                                    ],
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v6_autofit_IQR_exp-input",
                                    value=False,
                                    label="IQR exp",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Switch(
                                    id="ID_ABM_v6_autofit_IQR_model-input",
                                    value=False,
                                    label="IQR model",
                                )
                            ]
                        ),
                        dbc.Col(
                            width=1,
                            children=[
                                dbc.Button(
                                    "Download figure",
                                    id="ID_ABM_v6_autofit_download_figure-button",
                                    size="sm"
                                ),
                                dcc.Download("ID_ABM_v6_autofit_download"),
                            ]
                        ),
                    ]
                )
            ],
        )


        # Plot model results for given parameters
        @app.callback(
            Output("ID_ABM_v6_autofit_line-graph", "figure",allow_duplicate=True),
            Output("ID_ABM_v6_autofit_s-input","value"), 
            Output("ID_ABM_v6_autofit_RMSE-input","value"), 

            Input("ID_ABM_v6_autofit_runMultiple-input","n_clicks"),
            State("ID_ABM_v6_autofit_amountOfRuns-input","value"),

            State("ID_ABM_v6_autofit_N_begin-input", "value"),
            State("ID_ABM_v6_autofit_A_tpeak-input", "value"),
            State("ID_ABM_v6_autofit_A_tsigma-input", "value"),
            State("ID_ABM_v6_autofit_A_peak-input", "value"),
            State("ID_ABM_v6_autofit_slec_fraction-input", "value"),
            State("ID_ABM_v6_autofit_feedback_c-input", "value"),
            State("ID_ABM_v6_autofit_b_MPEC-input", "value"),
            State("ID_ABM_v6_autofit_b_SLEC-input", "value"),
            State("ID_ABM_v6_autofit_d_N-input", "value"),
            State("ID_ABM_v6_autofit_d_MPEC-input", "value"),
            State("ID_ABM_v6_autofit_d_SLEC-input", "value"),
            State("ID_ABM_v6_autofit_d_S-input", "value"),
            State("ID_ABM_v6_autofit_d_C-input", "value"),
            State("ID_ABM_v6_autofit_d_E-input", "value"),
            State("ID_ABM_v6_autofit_d_R-input", "value"),
            State("ID_ABM_v6_autofit_f_S-input", "value"),
            State("ID_ABM_v6_autofit_f_C-input", "value"),
            State("ID_ABM_v6_autofit_f_E-input", "value"),
            State("ID_ABM_v6_autofit_f_R-input", "value"),
            State("ID_ABM_v6_autofit_contraction_c-input", "value"),
            State("ID_ABM_v6_autofit_ren_E-input", "value"),

            State("ID_ABM_v6_autofit_IQR_exp-input","value"),
            State("ID_ABM_v6_autofit_IQR_model-input","value"),

            prevent_initial_call=True,
        )
        def getResultFromMultipleRuns(
            n_clicks,amountOfRuns,
            N_begin,
            A_tpeak,A_tsigma,A_peak,
            slec_fraction,
            feedback_c,
            b_MPEC,b_SLEC,
            d_N, d_MPEC, d_SLEC, d_S, d_C, d_E, d_R,
            f_S,f_C,f_E,f_R,
            contraction_c,
            ren_E,
            IQR_exp,IQR_model
        ):
            # Set parameters in the model
            self.model_abm_v6.N_begin=N_begin
            self.model_abm_v6.A_tpeak=A_tpeak
            self.model_abm_v6.A_tsigma=A_tsigma
            self.model_abm_v6.A_peak=A_peak
            self.model_abm_v6.slec_fraction=slec_fraction
            self.model_abm_v6.feedback_c=feedback_c
            self.model_abm_v6.b_MPEC=b_MPEC
            self.model_abm_v6.b_SLEC=b_SLEC
            self.model_abm_v6.d_N=d_N
            self.model_abm_v6.d_MPEC=d_MPEC
            self.model_abm_v6.d_SLEC=d_SLEC
            self.model_abm_v6.d_S=d_S
            self.model_abm_v6.d_C=d_C
            self.model_abm_v6.d_E=d_E
            self.model_abm_v6.d_R=d_R
            self.model_abm_v6.f_S=f_S
            self.model_abm_v6.f_C=f_C
            self.model_abm_v6.f_E=f_E
            self.model_abm_v6.f_R=f_R
            self.model_abm_v6.contraction_c=contraction_c
            self.model_abm_v6.ren_E=ren_E

            # Run the model multiple times
            data=self.model_abm_v6.simulateMultiple_MP(days=365,amount=amountOfRuns)

            dataMean=data.groupby(["Time_Point"])[self.TCell_Types_Responding].median()
        
            # Calculate global scaling factor s and normalized RMSE
            s,RMSE=calculate_s(dataMean)

            dataMean=dataMean*s
            # Calculate IQR for model results
            dataQ25=data.groupby(["Time_Point"])[self.TCell_Types_Responding].quantile(q=0.25)*s
            dataQ75=data.groupby(["Time_Point"])[self.TCell_Types_Responding].quantile(q=0.75)*s

            dataTimePoints=list(dataMean.index)

            #Get antigen
            antigen_time=np.array(dataTimePoints)
            antigen_values=dataMean[self.TCell_Types_Responding].max(axis=None)*np.exp(-0.5*((antigen_time-A_tpeak)/A_tsigma)**2)


            return getReplicationGraph(dataTimePoints,dataMean,dataQ25,dataQ75,IQR_exp,IQR_model,antigen_values),s,RMSE

        def getReplicationGraph(dataTimePoints,dataMean,dataQ25,dataQ75,
                                IQR_exp,IQR_model,
                                antigen_values):
            """Create a graph showing the model results"""
            fig=go.Figure()

            color_map = {
                        tcell: color
                        for tcell, color in zip(self.TCell_Types_Responding,
                                                px.colors.qualitative.Safe)
                    }
            
            #Plot model results and experimental data for each T cell type
            for TCell_Type in self.TCell_Types_Responding:
                data_exp_sub=self.expStats[self.expStats["TCell_Type"]==TCell_Type]
                color = color_map[TCell_Type]
                #Get model
                fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=dataMean[TCell_Type],
                    mode="lines",
                    line=dict(color=color),
                    name=f"{TCell_Type} model",
                    legendgroup=TCell_Type  # links subset model and experiment
                ))
                #Experimental data
                fig.add_trace(go.Scatter(
                    x=data_exp_sub["Time_Point"],
                    y=data_exp_sub["Median"],
                    mode="markers",
                    marker=dict(size=10,color=color),
                    name=f"{TCell_Type} exp",
                    legendgroup=TCell_Type,
                      
                    # IQR exp
                    error_y=dict(
                        type="data",
                        array=data_exp_sub["Q75"]-data_exp_sub["Median"],
                        arrayminus=data_exp_sub["Median"]-data_exp_sub["Q25"],

                        visible=IQR_exp
                    )
                ))
                # IQR model
                r, g, b = plotly.colors.unlabel_rgb(color)
                fig.add_trace(go.Scatter(
                    x=list(dataTimePoints)+list(dataTimePoints[::-1]),
                    y=list(dataQ25[TCell_Type].values)+list(dataQ75[TCell_Type].values[::-1]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    #fillcolor=color.replace("rgb", "rgba").replace(")", ",0.1)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=TCell_Type,
                    name=f"{TCell_Type} IQR",
                    visible=IQR_model,
                ))

            #Add antigen curve --- the plotted absolute values are not correct (it's just to show the shape of the curve)
            fig.add_trace(go.Scatter(
                    x=dataTimePoints,
                    y=antigen_values,
                    mode="lines",
                    line=dict(color="blue"),
                    #line=dict(color=color,dash="dash"),
                    name=f"Antigen",
                    #legendgroup=TCell_Type,  # links subset model and experiment
                    #showlegend=True,
                    #hoverinfo="skip",
                    visible="legendonly",
                ))
            
            fig.update_layout(
                template="plotly_white",
                title="Responses: model (lines) and experiment (dots)",
                xaxis_title="Time point (days)",
                yaxis_title="%AIM+ response",
                autosize=True
            )


            return fig

        def calculate_s(data):
            """Calculate the global scaling factor s and the normalized RMSE for the model results compared to the experimental data"""

            #Calculate numerator for s formula
            data_model=data.loc[[1,22,43,365]] #Returns relevant timepoints (the index represents the time_point in the dataframe)
            data_exp=self.expStats[self.expStats["TCell_Type"]!="Tnaive_response"].pivot_table(index="Time_Point",columns="TCell_Type",values="Median",sort=False) #Same form as data_model

            data_model=data_model.to_numpy()
            data_exp=data_exp.to_numpy()

            product_s=data_model*data_exp
            numerator_s=product_s.sum()

            #Calculate denominator
            square_s=data_model**2
            denominator_s=square_s.sum()

            #Calculate s
            s=numerator_s/denominator_s

            # Calculate normalized RMSE
            loss_matrix=((self.data_exp-s*data_model)/self.data_exp_normalization)**2
            SSE=loss_matrix.sum()
            RMSE=np.sqrt(SSE/self.number_of_datapoints)

            return s,RMSE

        #Changing visuals: show/hide IQR for experiment and model
        @app.callback(
            Output("ID_ABM_v6_autofit_line-graph", "figure",allow_duplicate=True),

            Input("ID_ABM_v6_autofit_IQR_exp-input","value"),
            Input("ID_ABM_v6_autofit_IQR_model-input","value"),

            State("ID_ABM_v6_autofit_line-graph", "figure"),

            prevent_initial_call=True,
        )
        def changeVisuals(IQR_exp,IQR_model,figOld):
            fig = go.Figure(figOld)
            
            visibilities={}
            for trace in fig.data:
                name=trace.name.split()
                if len(name)==1:
                    continue

                if name[1]=="model":
                    visibilities.update({name[0]:trace.visible})

                if name[1]=="IQR":
                    if IQR_model==False:
                        trace.visible=False
                    else:
                        trace.visible=visibilities[name[0]]

                if name[1]=="exp":
                    trace.error_y.visible=IQR_exp

            return fig


        # Importing a parameter solution found by the optimizer
        # Open the table containing the optimal solutions
        @app.callback(
            Output("ID_ABM_v6_autofit_Table_Modal","is_open"),
            Input("ID_ABM_v6_autofit_Load","n_clicks"),
            State("ID_ABM_v6_autofit_Table_Modal","is_open"),

            prevent_initial_call=True
        )
        def open_modal_table(n,is_open):
            return not is_open

        # Load the parameters of the selected row
        @app.callback(
            Output("ID_ABM_v6_autofit_Table_Modal","is_open",allow_duplicate=True),

            Output("ID_ABM_v6_autofit_N_begin-input", "value"),
            Output("ID_ABM_v6_autofit_A_tpeak-input", "value"),
            Output("ID_ABM_v6_autofit_A_tsigma-input", "value"),
            Output("ID_ABM_v6_autofit_A_peak-input", "value"),
            Output("ID_ABM_v6_autofit_slec_fraction-input", "value"),
            Output("ID_ABM_v6_autofit_feedback_c-input", "value"),
            Output("ID_ABM_v6_autofit_b_MPEC-input", "value"),
            Output("ID_ABM_v6_autofit_b_SLEC-input", "value"),
            Output("ID_ABM_v6_autofit_d_N-input", "value"),
            Output("ID_ABM_v6_autofit_d_MPEC-input", "value"),
            Output("ID_ABM_v6_autofit_d_SLEC-input", "value"),
            Output("ID_ABM_v6_autofit_d_S-input", "value"),
            Output("ID_ABM_v6_autofit_d_C-input", "value"),
            Output("ID_ABM_v6_autofit_d_E-input", "value"),
            Output("ID_ABM_v6_autofit_d_R-input", "value"),
            Output("ID_ABM_v6_autofit_f_S-input", "value"),
            Output("ID_ABM_v6_autofit_f_C-input", "value"),
            Output("ID_ABM_v6_autofit_f_E-input", "value"),
            Output("ID_ABM_v6_autofit_f_R-input", "value"),
            Output("ID_ABM_v6_autofit_contraction_c-input", "value"),
            Output("ID_ABM_v6_autofit_ren_E-input", "value"),

            Input("ID_ABM_v6_autofit_Table","selected_rows"),

            prevent_initial_call=True
        )
        def load_row(selected_rows):
            row=selected_rows[0]

            params=self.data_autofit_abm_v6["Parameters"][row].copy()
            #Parameter order
            order=["N_begin","A_tpeak","A_tsigma","A_peak","slec_fraction","feedback_c","b_MPEC","b_SLEC",
                    "d_N","d_MPEC","d_SLEC","d_S","d_C","d_E","d_R","f_S","f_C","f_E","f_R","contraction_c","ren_E"]
            params_list=[]
            for key in order:
                params_list.append(params[key])


            return tuple([False]+params_list)
            
        # Download the figure
        @app.callback(
            Output("ID_ABM_v6_autofit_download", "data"),

            Input("ID_ABM_v6_autofit_download_figure-button", "n_clicks"),

            State("ID_ABM_v6_autofit_line-graph","figure"),
            prevent_initial_call=True,
        )
        def save_plot(n,fig_data):
            fig=go.Figure(fig_data)
            return dcc.send_bytes(
                lambda buf: fig.write_image(buf, format="png", scale=4, width=1600, height=800),
                "Plot.png"
            )

        dash.register_page("ABM-v6-autofit", layout=layout,name="ABM v6 autofit")



if __name__=="__main__":
    # stats_object=Statistics()

    # print(stats_object.getResponders(TCell_Type="TSCM_response"))

    # stats_object.plotMedian()
    # stats_object.plotSpaghetti()
    # stats_object.plotResponders()
    # stats_object.plotPathwayBiasAnalysis()

    CompareModels_object=CompareModels()
    # CompareModels_object.ABM()
    # CompareModels_object.ABM_v5()
    # CompareModels_object.ABM_v6()
    # CompareModels_object.ABM_presentation()
    # CompareModels_object.ABM_v6_presentation()
    CompareModels_object.ABM_autofit()
    CompareModels_object.ABM_v5_autofit()
    CompareModels_object.ABM_v5_2_autofit()
    CompareModels_object.ABM_v6_autofit()

    #plotCompareCompartmentModel()

    plotResults()