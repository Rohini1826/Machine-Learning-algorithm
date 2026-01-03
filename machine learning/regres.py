import numpy as np
from sklearn.linear_model import LinearRegression
X=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
Y=np.array([40,55,52,65,70,68,78,85,82,90])
model=LinearRegression()
model.fit(X,Y)
m=model.coef_[0]
c=model.intercept_
print("Slope(m):",round(m,2))   
print("Intercept(c):",round(c,2))
X_new=np.array([[8]])
Y_pred=model.predict(X_new)
print("Predicted marks for 11 hours:",round(Y_pred[0],2))
