import pandas as pd
import numpy as np

confCases = pd.read_csv("../data/totalInfectedByStates.csv")
print("loaded.")

for column_name in confCases.columns:
    if column_name != "Date":
        cases_by_state = confCases[column_name]

    