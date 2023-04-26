import pickle
import pandas as pd
import numpy as np

from dscribe.descriptors.soap import SOAP

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.model_selection import KFold, cross_val_score


# load datasets
with open("superconduct.pkl",'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame({'atoms':data})
atoms = df['atoms'].to_numpy()
energies = np.array([Tc for atom in atoms])

# shuffle data
data = np.concatenate((atoms.reshape(-1,1),Tc.reshape(-1,1)),axis=1)
np.random.shuffle(data)
atoms = data[:,0]
energies = data[:,1]

#Dscribe SOAP
soap_params = {
    'rcut': 8.0,
    'nmax': 5,
    'lmax': 5,
    'species': ['Hg','Ba','Cu','O'],
    'periodic': True,
    'average': 'outer',
}

soap_outputs = SOAP(**soap_params).create(atoms)

#Data treating

X_train, X_test, y_train, y_test = train_test_split(soap_outputs, Tc, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(0.99999)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Model training

rf = RandomForestRegressor(n_estimators=200,random_state=42)
rf_params = {
    'max_depth': [None,2,4,10],
    'max_features': ['auto','sqrt','log2'],
    'min_samples_split': [2,6,10],
    'min_samples_leaf': [1,3,5],
}
grid_search = GridSearchCV(rf,rf_params,cv=5,verbose=1,n_jobs=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train,y_train)

best_rf = grid_search.best_estimator_

#model evaluation

fit_r2 = best_rf.score(X_train,y_train)
fit_rmse = np.sqrt(mean_squared_error(y_train,best_rf.predict(X_train)))
fit_mae = mean_absolute_error(y_train,best_rf.predict(X_train))

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(best_rf, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=5)
scores1 = cross_val_score(best_rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=5)
cross_rmse = np.mean([np.sqrt(abs(score)) for score in scores])
cross_mae = np.mean([abs(score) for score in scores1])

rmse = np.sqrt(mean_squared_error(y_test,best_rf.predict(X_test)))
mae = mean_absolute_error(y_test,best_rf.predict(X_test))
maxe = max_error(y_test,best_rf.predict(X_test))