###############################################
# Transformer class for custom transformation #
###############################################
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import sys
import math


class CCTransformer(BaseEstimator, TransformerMixin):
    def init(self):
        return None

    def fit(self, X, y = None):
        return self

    def dir_to_cos_sin(self,direct: str)->str:
        direct = direct.strip()
        mapping = { "E":0,
                    "N":math.pi/2,
                    "W":math.pi,
                    "S":math.pi*1.5}
        angle = 0
        for char in direct:
            try:
                angle += mapping[char]
            except KeyError:
                print("Direction character %s is not a possible direction to map" % char)
                sys.exit()
        angle /= len(direct)
        cos = round(math.cos(angle),2)
        sin = round(math.sin(angle),2)
        return "%s,%s" % (cos,sin)

    def transform(self, X, y = None):
        df = X

        #transform wind direction into angle in terms of the cos and sin for the angle
        df = df.astype({"Direction":str})
        df["Direction"] = df["Direction"].map(self.dir_to_cos_sin,na_action='ignore')
        df[["cos","sin"]] = df["Direction"].str.split(pat=",",n=1,expand=True)
        df = df.astype({"cos":float,"sin":float})
        
        columns = set(df.columns)
        to_keep = set(["Speed","cos","sin"])
        to_delete = list(columns-to_keep)
        if(to_delete):
            df = df.drop(to_delete,axis=1)

        return df
