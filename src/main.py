from lin_reg import Lin_reg
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../ProjWindTurbine.txt", sep=",")
    my_linear_regression = Lin_reg(df, "POW", 2)
    print(my_linear_regression.target)
    print(my_linear_regression.predict())
