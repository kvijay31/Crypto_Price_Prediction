import math
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ErrorAnalysis:
    def __init__(self, y_true, y_pred, X, non_window_columns: list=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        
        if non_window_columns is None:
            self.non_window_columns = []
        else:
            self.non_window_columns = non_window_columns
        
        self.window = self.X.drop(self.non_window_columns, axis=1)
        
        # frames
        self.std_frame = self.get_window_std()
    
    def get_window_std(self):
        std_series = self.window.std(axis=1)
        df = pd.DataFrame({"y_true": self.y_true, "y_pred": self.y_pred})
        df["window_std"] = std_series
        df["squared_error"] = (df["y_true"] - df["y_pred"])**2
        df["error"] = (df["y_true"] - df["y_pred"])
        df["asset"] = self.X["asset"]
        df["dt"] = pd.to_datetime(self.X["index"])
        return df
    
    def corr_std_error(self, assets=None, squared=True):
        if assets is not None:
            if isinstance(assets, str):
                assets = [assets]
            plot_df = self.std_frame[self.std_frame["asset"].isin(assets)].copy()
        else:
            plot_df = self.std_frame.copy()
        
        if squared:
            corr = plot_df.groupby('asset')[['squared_error', 'window_std']].corr()
        else:
            corr = plot_df.groupby('asset')[['error', 'window_std']].corr()
        corr = corr.iloc[0::2]["window_std"]
        corr.index = corr.index.get_level_values('asset')
        corr = pd.DataFrame(corr)
        corr.columns = ["correlation"]
        plot_df["absolute_error"] = abs(plot_df["error"])
        errors = plot_df.groupby('asset')[['squared_error', 'absolute_error', "y_true"]].mean()
        errors.columns = ["MSE", "MAE", "mean"]
        corr = corr.join(errors)
        std = plot_df.groupby('asset')[["y_true"]].std()
        std.columns = ["std"]
        corr = corr.join(std)
        overall_corr = plot_df[["squared_error", "window_std"]].corr().iloc[0]["window_std"]
        return corr, overall_corr
        
    def plot_std_error(self, assets=None, squared=True, width=700, height=500):
        """Standard deviation vs. squared error"""
        if assets is not None:
            if isinstance(assets, str):
                assets = [assets]
            plot_df = self.std_frame[self.std_frame["asset"].isin(assets)].copy()
            assets_string = ", ".join(assets)
        else:
            plot_df = self.std_frame.copy()
            assets_string = "all assets"
        
        if squared:
            error = "squared_error"
            error_string = "Squared error"
        else:
            error = "error"
            error_string = "error"
            
        fig = go.Figure()
        
        for asset_name in plot_df["asset"].unique():
            temp_df = plot_df[plot_df["asset"] == asset_name]
            fig.add_trace(go.Scatter(x=temp_df["window_std"], y=temp_df[error],
                        mode='markers',
                        name=asset_name))
        
        fig.update_layout(
            title=f"Standard deviation vs. {error_string} {assets_string}",
            xaxis_title="Window standard deviation",
            yaxis_title=error_string,
            legend_title="Assets",
            width=width,
            height=height,
            margin=dict(
                l=15,
                r=15,
                b=15,
                t=35,
                pad=7
            ),
        )
        return fig
    
    def error_over_time(self, assets=None, squared=True, width=1000, height=None):
        if assets is not None:
            if isinstance(assets, str):
                assets = [assets]
            plot_df = self.std_frame[self.std_frame["asset"].isin(assets)].copy()
            assets_string = ", ".join(assets)
        else:
            assets = list(self.std_frame["asset"].unique())
            plot_df = self.std_frame.copy()
            assets_string = "all assets"
        
        if squared:
            error = "squared_error"
            error_string = "Squared error"
        else:
            error = "error"
            error_string = "Error"
        
        devide = math.ceil(len(assets) / 2)
        fig = make_subplots(rows=devide, cols=2,
                    vertical_spacing=0.08, horizontal_spacing=0.1, subplot_titles=tuple(assets))
        
        rows = sorted(list(range(1, devide + 1)) * 2)
        columns = [1, 2] * devide
        for asset_name, row, col in zip(assets, rows, columns):
            temp_df = plot_df[plot_df["asset"] == asset_name].sort_values("dt")
            fig.add_trace(go.Scatter(x=temp_df["dt"], y=temp_df[error],
                        mode='lines',
                        name=asset_name), row=row, col=col)


        if height is None:
            height = min(250 * devide, 3000)
        fig.update_layout(
            title=dict(text=f"{error_string} over time for {assets_string}"),
            width=width,
            height=height,
            margin=dict(
                l=40,
                r=15,
                b=15,
                t=80,
                pad=4
            ),
        )
        
        y_min = min(plot_df[error]) * 0.8
        y_max = max(plot_df[error]) * 1.2
        for i in range(1, len(assets) + 1):
            fig['layout'][f'yaxis{i}'].update(title=error_string, range=[y_min, y_max], autorange=False, title_standoff=2)  
        return fig
    
    def predictions(self, assets=None, width=1000, height=None):
        if assets is not None:
            if isinstance(assets, str):
                assets = [assets]
            plot_df = self.std_frame[self.std_frame["asset"].isin(assets)].copy()
            assets_string = ", ".join(assets)
        else:
            assets = list(self.std_frame["asset"].unique())
            plot_df = self.std_frame.copy()
            assets_string = "all assets"
        
        devide = math.ceil(len(assets) / 2)
        fig = make_subplots(rows=devide, cols=2,
                    vertical_spacing=0.08, horizontal_spacing=0.1, subplot_titles=tuple(assets))
        
        rows = sorted(list(range(1, devide + 1)) * 2)
        columns = [1, 2] * devide
        for asset_name, row, col in zip(assets, rows, columns):
            temp_df = plot_df[plot_df["asset"] == asset_name].sort_values("dt")
            fig.add_trace(go.Scatter(x=temp_df["dt"], y=temp_df["y_true"],
                        mode='lines',
                        name=f"""{asset_name} Actual"""), row=row, col=col)
            
            fig.add_trace(go.Scatter(x=temp_df["dt"], y=temp_df["y_pred"],
                        mode='lines',
                        name=f"""{asset_name} Prediction"""), row=row, col=col)


        if height is None:
            height = min(250 * devide, 3000)
        fig.update_layout(
            title=dict(text=f"Predictions vs. actual over time for {assets_string}"),
            width=width,
            height=height,
            margin=dict(
                l=40,
                r=15,
                b=15,
                t=80,
                pad=4
            ),
        )
        
        y_min = min(min(plot_df["y_pred"]) * 0.9, min(plot_df["y_true"]) * 0.9)
        y_max = max(max(plot_df["y_pred"]) * 0.8, max(plot_df["y_true"]) * 0.8)
        for i in range(1, len(assets) + 1):
            fig['layout'][f'yaxis{i}'].update(title="Return", range=[y_min, y_max], autorange=False, title_standoff=2)  
        return fig
    
    def report(self, assets=None, squared=False, correlation_squared=True):
        corr_std_error_df, overall_corr = self.corr_std_error(assets=assets, squared=correlation_squared)
        fig_plot_std_error = self.plot_std_error(assets=assets, squared=squared)
        fig_error_over_time = self.error_over_time(assets=assets, squared=squared)
        
        print("-"*20, "Error report", "-"*20)
        print("Correlation of window standard deviation and error", overall_corr, " "*20)
        print(corr_std_error_df.T)
        print("* correlation: Pearson correlation of the standard deviation of the window returns and the (squared) error.")
        print(" ")
        print("-"*20, "Standard deviation and error", "-"*20)
        fig_plot_std_error.show(renderer="png")
        print(" ")
        print("-"*20, "Error over time per asset", "-"*20)
        fig_error_over_time.show(renderer="png")
        