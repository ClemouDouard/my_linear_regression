import pandas as pd
import numpy as np

class Lin_reg:

    def __init__(self, df: pd.DataFrame, target: str, step: float) -> None:
        self.target = np.array(df[target])
        self.init_coef()
        self.step = step
        self.values = np.array(df.drop(columns=[target]))
        self.values = np.c_[np.ones(self.values.shape[0]), self.values]
        self.n = df.shape[0]
        self.p = df.shape[1]

    def init_coef(self) -> None:
        self.coef = np.array([])
        self.coef = np.append(self.coef, np.mean(self.target))
        for i in range (1, self.p):
            self.coef = np.append(self.coef, 0)

    def predict(self) -> float:
        return np.dot(self.values, self.coef)

    def MSE(self) -> float:
        return 1/self.n * float(np.sum(self.target - self.predict()))**2

    def grad(self) -> np.ndarray:
        grad = np.array([])
        for i in range (self.p):
            grad = np.append(grad, -2/self.n * float(np.sum(np.multiply(self.values[:, i], self.target - self.predict()))))
        return grad
