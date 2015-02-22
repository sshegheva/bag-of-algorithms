import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd


def plot_waldo_coord(waldo_df, ax):
    plt.figure(figsize=(12.75, 8))
    #colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')
    #ax.set_color_cycle(colors)
    waldo_df.plot(x='X', y='Y', kind='scatter', ax=ax)
    ax.set_xlim(0, 12.75)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_waldo_kde(waldo_df, ax):
    plt.figure(figsize=(12.75, 8))
    sb.kdeplot(waldo_df.X, waldo_df.Y, shade=True, cmap="Blues", ax=ax)
    ax.set_xlim(0, 12.75)
    ax.set_ylim(0, 8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])