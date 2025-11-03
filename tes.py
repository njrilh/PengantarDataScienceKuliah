import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv')

print("========================== Data Titanic ==========================")
print(data)
print("=" * 50)

print("\n========================== Deskriptif ==========================")
print(data.describe())
print("=" * 50)

print("\n========================== Data Missing ==========================")
print(f'Data Missing :\n{data.isnull().sum()}')
print("\n========================== Data Duplikat ==========================")
print(f'Data Duplikat :\n{data.duplicated().sum()}')

print("=" * 50)
print(f"\nAge Maks = {data["age"].max()}")
print(f"\nAge Min = {data["age"].min()}")
print(f"\nAge Mean = {data["age"].mean()}")