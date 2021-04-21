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

# Other
import matplotlib.pyplot as plt
import pickle
plt.style.use('seaborn')

from transformer import CCTransformer

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

def eval_save_model(clf,x,y):
    # Load previous best model
    model_file = "best_model.pkl"
    try:
        with open(model_file,"rb") as infile:
            prev_model = pickle.load(infile)
        prev_score = round(prev_model.score(x,y),2)
    except FileNotFoundError:
        prev_score = 0
    new_score = round(clf.score(x,y),2)
    print("Old model score on test set:\t\t\t",prev_score)
    print("New model score on test set:\t\t\t",new_score)
    if(new_score > prev_score):
        with open(model_file,"wb") as outfile:
            pickle.dump(clf,outfile)
        return clf
    return prev_model

def predict_future(clf,show_plot=False,save_prediction=False):
    client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
    client.switch_database('orkney')

    # Future forecast
    forecasts  = client.query(
    "SELECT * FROM MetForecasts where time > now()"
    ) 
    for_df = get_df(forecasts)

    # Limit to only the newest source time
    newest_source_time = for_df["Source_time"].max()
    newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time].copy()

    gen_pred = clf.predict(newest_forecasts)
    if(show_plot):
        plt.plot(newest_forecasts.index,gen_pred)
        plt.xlabel("Time",fontweight=1000)
        plt.ylabel("Power generation (MW)",fontweight=1000)
        plt.title("Power generation prediction",fontsize=20,fontweight=1000)
        plt.show()
    if(save_prediction):
        out_df = pd.DataFrame(data={"time":newest_forecasts.index,"Prediction":gen_pred}).set_index("time")
        out_df.to_csv("Predictions.csv")

    return gen_pred

def make_north_plot(clf,x,y,plt_name):
    wind_dir = "N"
    speeds = [0.1*i for i in range(200)]
    dire = [wind_dir]*len(speeds)
    df = pd.DataFrame(data={"Speed":speeds,"Direction":dire})
    pred = clf.predict(df)
    plt.scatter(x[x["Direction"] == wind_dir]["Speed"],y[x["Direction"]==wind_dir])
    plt.plot(df["Speed"],pred,c='red')
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power generation (MW)")
    plt.title("Power generation estimation by Decision Tree Regressor")
    plt.savefig(plt_name)

########
# MAIN #
########
def main():
    ### HYPER PARAMS ###
    test_size = 0.30
    # Days to 'look into' the past for train-test data
    days = 120
    # Time span to look for when down sampling, 
    # e.g. 60 min prior to 60 min after, every third hour
    time_span = 60
    
    ### DATA PREPARATION FOR TRAINING ###
    x,y = prepare_data(days=days,span = time_span)

    ### MODEL TRAINING ###
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,shuffle=False)
    clf = train_new(x_train,y_train)
    
    ### EVALUATING BEST MODEL ###
    clf = eval_save_model(clf,x_test,y_test)

    #Unmark this to see feature importances according to model
    #print(clf.named_steps["Estimator"].feature_importances_)

    ### PREDICT FROM LATEST FORECAST ###
    predicted = predict_future(clf,show_plot=False,save_prediction=False)

    # Uncomment this to show a plot of the fitted model predictions
    # Mostly used for debugging and evalutating
    #make_north_plot(clf,x_train,y_train,plt_name="./Images/NORTH_DT.png")


if __name__ == "__main__":
    main()