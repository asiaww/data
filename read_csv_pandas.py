import pandas as pd

benign_df = pd.read_csv("./negative_list.csv")
print(benign_df.head(5))
print(benign_df.count)
