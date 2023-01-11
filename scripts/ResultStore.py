import os
import pandas as pd
import plotly.graph_objects as go

class ResultStore:
    def __init__(self, load_if_exists=False, path=".", name=None):
        if load_if_exists:
            self.load(path, name)
        else:
            self.data = []
        
    def add(self, model, scores, predictions, y_validate, meta=None):
        data = {"model": str(type(model).__name__), "predictions": predictions, "truth": y_validate, "meta": meta}
        data.update(scores)
        self.data.append(data)
    
    def get_df(self):
        return pd.DataFrame(self.data)
    
    def load(self, path=".", name=None):
        if name is None:
            name = "results.json"
        df = pd.read_json(os.path.join(path, name))
        self.data = df.to_dict('records')
    
    def save(self, path=".", name=None):
        if name is None:
            name = "results.json"
        self.get_df().to_json(os.path.join(path, name))
        
    def plot(self, row=None):
        df = self.get_df()
        fig = go.Figure()
        for index, data in df.iterrows():
            if row is not None:
                if index not in row:
                    continue
            fig.add_trace(go.Scatter(x=list(range(len(data["truth"]))), y=data["truth"],
                    mode='markers',
                    name=f"""{data["model"]} True"""))
            fig.add_trace(go.Scatter(x=list(range(len(data["predictions"]))), y=data["predictions"],
                    mode='markers',
                    name=f"""{data["model"]} Prediction"""))
            
        return fig
    
    