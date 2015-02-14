import matplotlib.pyplot as plt
import seaborn as sb


def plot_waldo_coord(waldo_df):
    plt.figure(figsize=(12.75, 8))
    plt.plot([6.375, 6.375], [0, 8], "--", color="black", alpha=0.4, lw=1.25)

    for book, group in waldo_df.groupby("Book"):
        plt.plot(group.X, group.Y, "o", label="Book %d" % (book))

    plt.xlim(0, 12.75)
    plt.ylim(0, 8)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper center", ncol=7, frameon=True, fancybox=True, bbox_to_anchor=(0.5, 1.1))


def plot_waldo_kde(waldo_df):
    plt.figure(figsize=(10, 6))
    sb.kdeplot(waldo_df.X, waldo_df.Y, shade=True, cmap="Blues")
    plt.xlim(0, 12.75)
    plt.ylim(0, 8)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])