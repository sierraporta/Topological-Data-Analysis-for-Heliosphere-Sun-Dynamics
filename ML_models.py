import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("datas/dataforR_all.csv",index_col=0)
var="R"
#data = pd.read_csv("datas/dataforDst_all.csv",index_col=0)
#var="Dst"
#data = pd.read_csv("datas/dataforF10_all.csv",index_col=0)
#var="f107"
data=data[data.index>="2008-12"] # Only cycl2 24 and 25
data=data.rename(columns={"R (Sunspot)":"R",
                          "Dst-index":"Dst",
                          "f10.7_index":"f107",
                          "Scalar B, nT":"B",
                          "SW Plasma Temperature, K":"swT",
                          "SW Proton Density, N/cm^3":"swN",
                          "SW Plasma Speed, km/s":"swV",
                          "Alpha/Prot. ratio":"APr"})

from sklearn import preprocessing
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
dataq = pd.DataFrame(quantile_transformer.fit_transform(data),index=data.index,columns=data.columns)
data=dataq.copy()

N=-1
X = data.drop([var], axis=1)[1:N]
X1 = data.drop([var,"B","swT","swN","swV","APr"], axis=1)[1:N]
X2 = data.drop([var,"Shannon Entropy","Sample Entropy","Permutation Entropy",
                "Spectral Entropy","Approximate Entropy","Higuchi Fractal Dim.",
                "Katz Fractal Dim.","Petrosian Fractal Dim.","Lempel-Ziv Complexity","Hurst Exponent"], axis=1)[1:N]
y = data[var][1:N]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

import tpot
from tpot import TPOTRegressor

#########################
GENERATIONS = 6
POPULATION = 10
CROSSVALIDATION_SPLIT = 4
#########################
tpot = TPOTRegressor(verbose=4,
                     #max_time_mins=10,
                     n_jobs=4,
                     generations=GENERATIONS,
                     cv=CROSSVALIDATION_SPLIT)
tpot.fit(X_train, y_train)

print(tpot.fitted_pipeline_)

tpot.fitted_pipeline_.steps[-1][1]

# Get the best model
exctracted_best_model = tpot.fitted_pipeline_.steps[-1][1]

import pickle
# save the model to disk
filename = 'datas/finalized_model_tpot_cycle_24_25_R.sav'
pickle.dump(exctracted_best_model, open(filename, 'wb'))


