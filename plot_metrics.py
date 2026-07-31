import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics.csv", help="Ścieżka do pliku z metrykami (CSV)")
    ap.add_argument("--save", default="metrics_plot.png", help="Ścieżka do pliku wynikowego (.png)")
    ap.add_argument("--window", type=int, default=25, help="Rozmiar okna wygładzania dla średnich ruchomych")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df = df.sort_values("episode")

    win = max(1, args.window)
    df["avg_score_smooth"] = smooth(df["avg_all"], win)
    df["loss_smooth"] = smooth(df["loss"], win)
    df["epsilon_smooth"] = smooth(df["epsilon"], win)
    df["lr_smooth"] = smooth(df["lr"], win)

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 7))

    axs[0, 0].plot(df["episode"], df["avg_score_smooth"], label="avg score (smooth)")
    axs[0, 0].set_title("Average score")
    axs[0, 0].set_ylabel("avg_all")

    axs[0, 1].plot(df["episode"], df["epsilon_smooth"], color="tab:orange")
    axs[0, 1].set_title("Epsilon")
    axs[0, 1].set_ylabel("epsilon")

    axs[1, 0].plot(df["episode"], df["loss_smooth"], color="tab:red")
    axs[1, 0].set_title("Loss (per episode)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("loss_sum")

    axs[1, 1].plot(df["episode"], df["lr_smooth"], color="tab:green")
    axs[1, 1].set_title("Learning rate")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("lr")

    for ax in axs.flat:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    print(f"Saved plot to {args.save}")


if __name__ == "__main__":
    main()
