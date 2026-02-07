from data_loader import load_data

data = load_data("KNNeighborsML/data/Social_Network_Ads.csv")
print(data.head())

data2 = load_data("KNNeighborsML/data/multiple_regression_age.csv")
print(data2.head())