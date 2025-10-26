import pandas as pd
import numpy as np

class LinearRegression:
    
    def __init__(self, n_iter, learning_rate):
        self.iteration = n_iter
        self.weight = None
        self.bias = 0
        self.learning_rate = learning_rate
        
    def predict(self, x):
        return np.dot(x ,self.weight) + self.bias
    
    def fit(self, x_data, y_data):
        self.feauters = x_data.to_numpy()
        self.actual = y_data.to_numpy()
        
        self.weight = np.zeros(self.feauters.shape[1])
        
        for i in range(self.iteration):
            self.y_pred = self.predict(self.feauters)
            
            gradient_w = (1/self.feauters.shape[0]) * np.sum(np.dot(self.feauters.T, (self.y_pred-self.actual)))
            gradient_b = (1/self.feauters.shape[0]) * np.sum(self.y_pred-self.actual)
            
            
            self.weight = self.weight - self.learning_rate * gradient_w
            self.bias = self.bias - self.learning_rate * gradient_b
            

class StandardScalar:
    def __init__(self):
        self.mean = 0
        self.standard = 0
        
    def val_calc(self, data):
        self.mean = np.mean(data, axis=0)
        self.standard = np.std(data, axis=0)
        
        return self.mean, self.standard
        
    def fit_transform(self, data):
        
        self.mean,self.standard = self.val_calc(data)
        return (data-self.mean) / self.standard
    
    def transform(self, data):
        return (data-self.mean) / self.standard
          

data = pd.read_csv('Salary_dataset.csv')
train_data = data[:26]
test_data = data[26:]

x_train,y_train = train_data.drop(['Salary'], axis=1), train_data['Salary']
x_test,y_test = test_data.drop(['Salary'], axis=1), test_data['Salary']


scalar = StandardScalar()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

model = LinearRegression(700, 0.01)
model.fit(x_train,y_train)

prediction = model.predict(x_test)

rmse = ((1/prediction.shape[0]) * np.sum((y_test-prediction) ** 2)) ** 0.5


print("Root mean Squared Error: ",rmse)