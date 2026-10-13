import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from lost_found_matcher import load_and_clean, LostFoundMatcher, evaluate, MIN_SCORE

STRONG_SCORE = 0.70  # same cutoff app.py uses for the "Strong" label
OUTPUT = "cosine_vs_knn_chart.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
METHODS = [
    ("cosine", "Cosine similarity (primary)", "#2a78d6"),
    ("knn", "K-Nearest Neighbors (comparison)", "#eb6834"),
]


def swarm_offsets(values, step=0.07, min_gap=0.03):
    """Spread dots sideways so close scores don't hide each other."""
    offsets = np.zeros(len(values))
    placed = []
    for i in np.argsort(values):
        for x in [0, -step, step, -2 * step, 2 * step, -3 * step, 3 * step]:
            if all(abs(values[i] - y) >= min_gap or px != x for y, px in placed):
                break
        offsets[i] = x
        placed.append((values[i], x))
    return offsets


def draw_panel(ax, res, title, color):
    has_match = res[res["true_match"] != "-"]
    no_match = res[res["true_match"] == "-"]
    groups = [
        (0, has_match, f"Has a true match\n({len(has_match)} items)"),
        (1, no_match, f"No true match\n({len(no_match)} items)"),
    ]

    for y, label in [(STRONG_SCORE, "Strong 0.70"), (MIN_SCORE, "Cutoff 0.40")]:
        ax.axhline(y, color=INK_2 if y == MIN_SCORE else MUTED, lw=1, zorder=1)
        ax.text(
            1.5, y + 0.012, label, ha="right", va="bottom", fontsize=8.5, color=INK_2
        )

    means = []
    for x, group, _ in groups:
        scores = group["top1_score"].to_numpy()
        xs = x + swarm_offsets(scores)
        correct = (group["top1"] == group["true_match"]).to_numpy()
        # No-match items have no "correct" answer, so they are drawn filled like hits.
        filled = correct | (group["true_match"] == "-").to_numpy()
        ax.scatter(
            xs[filled],
            scores[filled],
            s=64,
            color=color,
            edgecolors=SURFACE,
            linewidths=2,
            zorder=3,
        )
        ax.scatter(
            xs[~filled],
            scores[~filled],
            s=64,
            facecolors=SURFACE,
            edgecolors=color,
            linewidths=2,
            zorder=3,
        )

        mean = scores.mean()
        means.append(mean)
        ax.plot(
            [x - 0.3, x + 0.3],
            [mean, mean],
            color=INK,
            lw=2,
            solid_capstyle="round",
            zorder=4,
        )
        ax.text(
            x + 0.33,
            mean,
            f"avg {mean:.2f}",
            va="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )

        # Label only the dots the write-up talks about.
        for item_id in ["R013", "R025"]:
            row = group.index[group["reported_id"] == item_id]
            if len(row):
                i = group.index.get_loc(row[0])
                ax.annotate(
                    item_id,
                    (xs[i], scores[i]),
                    xytext=(-8, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    color=INK_2,
                )

    ax.set_title(
        f"{title}\nGap between averages: {means[0] - means[1]:.2f}",
        loc="left",
        fontsize=10.5,
        color=INK,
        linespacing=1.5,
    )
    ax.set_xticks([x for x, _, _ in groups], [lab for _, _, lab in groups])
    ax.set_xlim(-0.5, 1.55)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=9)


def main():
    df, _ = load_and_clean()
    matcher = LostFoundMatcher(df)

    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.08, right=0.97, wspace=0.12)

    for ax, (method, title, color) in zip(axes, METHODS):
        ax.set_facecolor(SURFACE)
        res, _ = evaluate(matcher, method)
        draw_panel(ax, res, title, color)
    axes[0].set_ylabel("Score of the #1 ranked match", color=INK_2, fontsize=9.5)

    fig.text(
        0.08,
        0.95,
        "Cosine separates real matches from non-matches more clearly",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.905,
        "Top-1 score for each of the 30 reported items. Both methods rank items "
        "the same way (87.5% top-1, 100% top-3);\nonly the score scale differs.",
        fontsize=9.5,
        color=INK_2,
        va="top",
        linespacing=1.4,
    )

    key = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=8,
            mfc=INK_2,
            mec=INK_2,
            label="Top-1 score of one reported item",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=8,
            mfc=SURFACE,
            mec=INK_2,
            mew=2,
            label="#1 was a look-alike (true match ranked #2)",
        ),
        Line2D([], [], color=INK, lw=2, label="Group average"),
    ]
    fig.legend(
        handles=key,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.015),
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        handletextpad=0.4,
        columnspacing=1.8,
    )

    fig.savefig(OUTPUT, dpi=200, facecolor=SURFACE)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
