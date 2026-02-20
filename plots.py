import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_file="results.csv"):

    df = pd.read_csv(csv_file)

    # Score vs Depth
    plt.figure()
    plt.scatter(df["depth"], df["score"])
    plt.xlabel("Circuit Depth")
    plt.ylabel("Score")
    plt.title("Score vs Circuit Depth")
    plt.show()

    # Loss vs CNOT
    plt.figure()
    plt.scatter(df["cnot"], df["loss"])
    plt.xlabel("CNOT Count")
    plt.ylabel("Loss")
    plt.title("Loss vs CNOT Count")
    plt.show()

    print("\nCorrelation Matrix:")
    print(df[["loss", "depth", "cnot", "score"]].corr())