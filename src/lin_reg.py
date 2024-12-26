import pandas as pd
import numpy as np

class Lin_reg:

    def __init__(self, df: pd.DataFrame, target: str, step: float) -> None:
        self.n = df.shape[0]
        self.p = df.shape[1]
        self.target = np.array(df[target])
        self.init_coef()
        self.step = step
        self.values = np.array(df.drop(columns=[target]))
        self.values = np.c_[np.ones(self.values.shape[0]), self.values]

    def init_coef(self) -> None:
        self.coef = np.zeros(self.p)


    def predict(self) -> float:
        return np.dot(self.values, self.coef)

    def MSE(self) -> float:
        return float(np.mean((self.target - self.predict()) ** 2))


    def grad(self) -> np.ndarray:
        errors = self.target - self.predict()
        grad = -2 / self.n * np.dot(self.values.T, errors)
        return grad


    def fit(self, max_iter: int, tol: float = 1e-6) -> None:
        prev_mse = float('inf')
        for i in range(max_iter):
            self.coef = self.coef - self.step * self.grad()
            current_mse = self.MSE()
            if abs(prev_mse - current_mse) < tol:
                print("algorithm converged")
                break
            prev_mse = current_mse
        print("maximum iteration reached")
