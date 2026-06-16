import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

# Class distribution
class_counts = df["Class"].value_counts()

print(class_counts)

# Plot
class_counts.plot(kind="bar")

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()