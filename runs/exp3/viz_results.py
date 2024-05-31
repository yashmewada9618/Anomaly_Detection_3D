import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "val_exp3.csv"
df = pd.read_csv(csv_file)

csv_train = "train_ecp3.csv"
df_train = pd.read_csv(csv_train)

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(df["Step"], df["Value"], label="validation Loss", marker="o")

# Adding labels and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("val_loss_exp3.png")

# Display the plot
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_train["Step"], df_train["Value"], label="Training Loss", marker="o")

# Adding labels and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("train_loss_exp3.png")
plt.show()
