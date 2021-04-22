#########################
#       IMPORTS         #
#########################

# Data gathering a manipulation
from influxdb import InfluxDBClient
import pandas as pd
import numpy as np

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import r2_score

# Other
import pickle
import mlflow

###############################################
# Transformer class for custom transformation #
###############################################
class CCTransformer(BaseEstimator, TransformerMixin):
    def init(self):
        return None

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        df = X

        columns = set(df.columns)
        to_keep = set(["Speed"])
        to_delete = list(columns-to_keep)
        if(to_delete):
            df = df.drop(to_delete,axis=1)

        return df

####################
# HELPER FUNCTIONS #
####################
def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index
    return df

def down_sample(df,span=60):
    t0 = df.index[0]
    t_end = df.index[-1] + pd.to_timedelta(1,unit='days')
    time_periods = pd.date_range(start=t0,end=t_end,freq='3H',normalize=True)
    means = []
    for time in time_periods:
        t_start = time - pd.to_timedelta(span,unit="minutes")
        t_end = time + pd.to_timedelta(span,unit="minutes")
        m = df[t_start : t_end].mean()
        if(len(df[t_start : t_end]) < 60):
            means.append(np.nan)
        else:
            means.append(m[0])
    sampled_df = pd.DataFrame(data = {"Produced":means,"time":time_periods}).set_index("time")
    return sampled_df

def prepare_data(days=90,span=60):
    # Loads data from orkney
    client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
    client.switch_database('orkney')

    # Get the last 'days' days of power generation data
    generation = client.query(
        "SELECT * FROM Generation where time > now()-%sd" % days
        )
    # Get the last 'days' days of weather forecasts with the shortest lead time
    wind  = client.query(
        "SELECT * FROM MetForecasts where time > now()-%sd and time <= now() and Lead_hours = '1'" % days
        )

    # Drop uneccessary columns
    gen_df = get_df(generation).drop(["ANM","Non-ANM"],axis=1)
    gen_df.rename(columns = {"Total" : "Produced"},inplace=True)
    # Downsample gen_df into 3 hour windows to fit with wheather forecast
    grouped_gen_df = down_sample(gen_df,span=span)
    #grouped_gen_df = gen_df.resample('3H').mean()

    # Get wind forecast data
    wind_df = get_df(wind)

    # Join to ensure all inputs have a label
    final_df  = wind_df.join(grouped_gen_df,how='inner').dropna()
    #tmp_df = wind_df.join(tmp,how="inner").dropna()

    x = final_df.iloc[:,:-1]
    y = final_df.iloc[:,-1]

    return x,y

def train_new(x,y):
    # Pipeline estimator
    # All preproccessing is done by custom transformer
    pipeline = Pipeline([
        ("custom_transformer",CCTransformer()),
        ("Estimator", DTR(min_samples_split=10))
    ])

    # Training
    pipeline.fit(x,y)
    #print("Mean cross-validated score of estimator:\t",round(grid.best_score_,2))
    return pipeline


########
# MAIN #
########
def main(test_size=0.3,days=120,time_span=60):
    # Room for altering parameters when testing things

    with mlflow.start_run() as run:
        mlflow.log_params({'test_size':test_size,
                          'days': days,
                          'time_span':time_span})

        ### DATA PREPARATION FOR TRAINING ###
        x,y = prepare_data(days=days,span = time_span)

        ### MODEL TRAINING ###
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,shuffle=False)
        clf = train_new(x_train,y_train)
        
        ### EVALUATING  ###
        r2 = r2_score(y_test,clf.predict(x_test))
        mlflow.log_metric('r2',r2)
        print('NEW MODEL SCORE: %s' % r2)

        mlflow.sklearn.log_model(clf,'model')

        return clf, r2, x_test, y_test, run.info.run_id


if __name__ == "__main__":
    main()