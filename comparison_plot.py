import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_comparison(random_scores, evolution_scores, llm_scores,
                    save_path="comparison_plot.png"):
    """
    Plots best-score-so-far convergence curves for all three search strategies.

    X-axis: cumulative circuit evaluation number
    Y-axis: best hardware-efficiency score found so far (lower = better)

    Args:
        random_scores:    list of floats from run_search()
        evolution_scores: list of floats from run_evolution_search()
        llm_scores:       list of floats from run_llm_search()
        save_path:        file to save the figure to
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Plot convergence curves ──────────────────────────────────────────
    ax.plot(
        range(1, len(random_scores) + 1), random_scores,
        marker="o", linewidth=2, markersize=5,
        label="Random Search", color="#e74c3c", linestyle="--"
    )
    ax.plot(
        range(1, len(evolution_scores) + 1), evolution_scores,
        marker="s", linewidth=2, markersize=5,
        label="Evolutionary Search", color="#f39c12", linestyle="-."
    )
    ax.plot(
        range(1, len(llm_scores) + 1), llm_scores,
        marker="^", linewidth=2.5, markersize=6,
        label="LLM-Guided Search", color="#2ecc71", linestyle="-"
    )

    # ── Annotate final best values ───────────────────────────────────────
    ax.annotate(f"{random_scores[-1]:.4f}",
                xy=(len(random_scores), random_scores[-1]),
                xytext=(5, 4), textcoords="offset points",
                fontsize=8, color="#e74c3c")
    ax.annotate(f"{evolution_scores[-1]:.4f}",
                xy=(len(evolution_scores), evolution_scores[-1]),
                xytext=(5, -12), textcoords="offset points",
                fontsize=8, color="#f39c12")
    ax.annotate(f"{llm_scores[-1]:.4f}",
                xy=(len(llm_scores), llm_scores[-1]),
                xytext=(5, 4), textcoords="offset points",
                fontsize=8, color="#2ecc71")

    # ── Labels and style ─────────────────────────────────────────────────
    ax.set_xlabel("Circuit Evaluations (cumulative)", fontsize=12)
    ax.set_ylabel("Best Score So Far  (lower = better)", fontsize=12)
    ax.set_title("VQC Architecture Search — Strategy Comparison\n"
                 "Score = Loss + λ·Depth + λ·CNOT", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[Plot] Saved to: {save_path}")
    plt.show()
