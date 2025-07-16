#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import subprocess
import re
from importlib.metadata import version

import zarr
import zarrs
import dask
import tensorstore

plt.rcParams['svg.hashsalt'] = 'deterministic'

LEGEND_COLS = 2
YMAX_READ_ALL = 4
YMAX_READ_ALL_DASK = 40
YMAX_READ_CHUNKS = 4
YMAX_ROUNDTRIP = 20
YMAX_ROUNDTRIP_DASK = 50
# YMAX_READ_ALL = None
# YMAX_READ_CHUNKS = None
# YMAX_ROUNDTRIP = None

# Get the implementation versions
if not hasattr(zarrs, "__version__"):
    zarrs.__version__ = version("zarrs")
if not hasattr(tensorstore, "__version__"):
    tensorstore.__version__ = version("tensorstore")
zarrs_tools_ver = subprocess.run(["zarrs_reencode", "--version"], stdout=subprocess.PIPE, text=True).stdout
if m := re.search(r"\(zarrs (.+?)\)", zarrs_tools_ver):
    # E.g. zarrs_tools 0.6.0-beta.1 (zarrs 0.18.0-beta.0) -> 0.18.0-beta.0
    zarrs_ver = m.group(1)
else:
    zarrs_ver = "0.18.0-beta.0"

IMPLEMENTATIONS = {
    "zarrs_rust": f"zarrs/zarrs ({zarrs_ver})",
    "tensorstore_python": f"google/tensorstore ({tensorstore.__version__})",
    "zarr_python": f"zarr-developers/zarr-python ({zarr.__version__})",
    "zarrs_python": f"zarr-developers/zarr-python ({zarr.__version__}) \n + zarrs/zarrs-python ({zarrs.__version__}) ZarrsCodecPipeline",
    "zarr_dask_python": "Default BatchedCodecPipeline",
    "zarrs_dask_python": f"ZarrsCodecPipeline via zarrs/zarrs-python ({zarrs.__version__})",
}

IMAGES = {
    "data/benchmark.zarr": "Uncompressed",
    "data/benchmark_compress.zarr": "Compressed",
    "data/benchmark_compress_shard.zarr": "Compressed\n + Sharded",
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["lmodern"],
    # "axes.autolimit_mode": "round_numbers",
})

def custom_bar_label(ax, padding=5, rotation=90):
    """Adds labels to bars in a bar chart.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axes containing the bars.
        padding (int): Padding for the labels.
        rotation (int): Rotation angle for the labels.
    """
    y_lim = ax.get_ylim()[1]  # Get the upper limit of the y-axis

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            # Determine label position based on whether the bar exceeds y-axis limit
            label_position = min(height, y_lim)  # Use y_lim if height exceeds it
            ax.annotate(f'{height:.3g}', 
                        xy=(bar.get_x() + bar.get_width() / 2, label_position),
                        xytext=(0, padding),
                        textcoords="offset points",
                        ha='center', 
                        va='bottom',
                        rotation=rotation,
                        clip_on=False)

def plot_read_all(plot_dask: bool, ymax: float):
    df = pd.read_csv("measurements/benchmark_read_all.csv", header=[0, 1], index_col=0)
    df.index = ["Uncompressed", "Compressed", "Compressed\n+ Sharded"]

    if plot_dask:
        df = df.loc[:, df.columns.get_level_values(1).str.contains("dask")]
    else:
        df = df.loc[:, ~df.columns.get_level_values(1).str.contains("dask")]

    df.rename(level=1, columns=IMPLEMENTATIONS, inplace=True)
    print(df)


    # Prepare split axis figure and axes
    fig = plt.figure(figsize=(9, 4), layout="constrained")
    spec = fig.add_gridspec(2, 2)
    ax_time = fig.add_subplot(spec[:, 0])
    ax_mem = fig.add_subplot(spec[:, 1])

    # Plot the data
    df["Time (s)"].plot(kind='bar', ax=ax_time)
    ax_time.set_ylim(ymin=0)
    title = f"dask/dask ({dask.__version__}) + zarr-developers/zarr-python ({zarr.__version__})" if plot_dask else "Zarr V3 Implementation"
    fig.legend(loc='outside upper center', ncol=LEGEND_COLS, title=title, borderaxespad=0)
    df["Memory (GB)"].plot(kind='bar', ax=ax_mem)

    # Styling
    ax_time.set_ylabel("Elapsed time (s)")
    ax_time.set_ylim(ymin=0, ymax=ymax)
    ax_time.tick_params(axis='x', labelrotation=0)
    ax_time.grid(True, which='both', axis='y')
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)
    ax_mem.set_ylabel("Peak memory usage (GB)")
    ax_mem.tick_params(axis='x', labelrotation=0)
    ax_mem.grid(True, which='both', axis='y')
    ax_mem.spines['top'].set_visible(False)
    ax_mem.spines['right'].set_visible(False)

    custom_bar_label(ax_time)
    custom_bar_label(ax_mem)

    ax_time.get_legend().remove()
    ax_mem.get_legend().remove()

    fig.savefig(f"plots/benchmark_read_all{'_dask' if plot_dask else ''}.svg", metadata={'Date': None, 'Creator': None})
    # fig.savefig(f"plots/benchmark_read_all{'_dask' if plot_dask else ''}.pdf", metadata={'Date': None, 'Creator': None})


def plot_read_chunks(plot_dask: bool):
    df = pd.read_csv("measurements/benchmark_read_chunks.csv", header=[0, 1], index_col=[0, 1])

    if plot_dask:
        df = df.loc[:, df.columns.get_level_values(1).str.contains("dask")]
    else:
        df = df.loc[:, ~df.columns.get_level_values(1).str.contains("dask")]

    # Reduce the dictionary to contain only the specified keys
    def reduce_dict(d, keys):
        return {k: d[k] for k in keys if k in d}

    implementations = reduce_dict(IMPLEMENTATIONS, df.columns.get_level_values(1))

    df = df.reset_index(level=1)
    print(df)

    fig = plt.figure(figsize=(9, 4), layout="constrained")
    spec = fig.add_gridspec(2, 2)
    ax_time = fig.add_subplot(spec[:, 0])
        
    ax_mem = fig.add_subplot(spec[:, 1])

    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    image_ls = {'data/benchmark.zarr': ":", 'data/benchmark_compress.zarr': '--', 'data/benchmark_compress_shard.zarr': '-'}
    for image, row in df.groupby("Image"):
        row.plot(x="Concurrency", y="Time (s)", ax=ax_time, color=cmap, ls=image_ls[image])
        row.plot(x="Concurrency", y="Memory (GB)", ax=ax_mem, color=cmap, ls=image_ls[image])

    # Custom legend
    custom_lines = [Line2D([0], [0], color=cmap[i]) for i in range(len(implementations))]
    title = f"dask/dask ({dask.__version__}) + zarr-developers/zarr-python ({zarr.__version__})" if plot_dask else "Zarr V3 Implementation"
    fig.legend(custom_lines, [implementation.replace(" ", " ") for implementation in implementations.values()], loc="outside upper left", ncol=2, title=title, borderaxespad=0)
    custom_lines = [Line2D([0], [0], color='k', ls=':'),
                Line2D([0], [0], color='k', ls='--'),
                Line2D([0], [0], color='k', ls='-')]
    fig.legend(custom_lines, IMAGES.values(), loc="outside upper right", ncol=2, title="Dataset", borderaxespad=0)

    ax_time.get_legend().remove()
    ax_mem.get_legend().remove()

    ax_time.set_ylabel("Elapsed time (s)")

    xticks = [1, 2, 4, 8, 16, 32]
    ax_time.set_ylim(ymin=0, ymax=YMAX_READ_CHUNKS)
    ax_time.set_xscale('log', base=2)
    ax_time.xaxis.set_major_formatter(plt.FuncFormatter("{:.0f}".format))
    ax_time.set_xlim(1, 32) 
    ax_time.set_xticks(xticks)
    ax_time.set_xlabel("Concurrent chunks")
    ax_time.grid(True, which='both', axis='y')
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)

    ax_mem.set_yscale('log')
    ax_mem.set_xscale('log', base=2)
    ax_mem.xaxis.set_major_formatter(plt.FuncFormatter("{:.0f}".format))
    ax_mem.set_xlim(1, 32)
    ax_mem.set_xticks(xticks)
    ax_mem.set_xlabel("Concurrent chunks")
    ax_mem.set_ylabel("Peak memory usage (GB)")
    ax_mem.grid(True, which='both', axis='y')
    ax_mem.spines['top'].set_visible(False)
    ax_mem.spines['right'].set_visible(False)

    custom_bar_label(ax_time)
    custom_bar_label(ax_mem)

    fig.savefig(f"plots/benchmark_read_chunks{'_dask' if plot_dask else ''}.svg", metadata={'Date': None, 'Creator': None})
    # fig.savefig(f"plots/benchmark_read_chunks{'_dask' if plot_dask else ''}.pdf", metadata={'Date': None, 'Creator': None})


def plot_roundtrip(plot_dask: bool, ymax: float):
    df = pd.read_csv("measurements/benchmark_roundtrip.csv", header=[0, 1], index_col=0)
    df.index = ["Uncompressed", "Compressed", "Compressed\n+ Sharded"]

    if plot_dask:
        df = df.loc[:, df.columns.get_level_values(1).str.contains("dask")]
    else:
        df = df.loc[:, ~df.columns.get_level_values(1).str.contains("dask")]

    df.rename(level=1, columns=IMPLEMENTATIONS, inplace=True)
    print(df)

    # Prepare split axis figure and axes
    fig = plt.figure(figsize=(9, 4), layout="constrained")
    spec = fig.add_gridspec(2, 2)
    ax_time = fig.add_subplot(spec[:, 0])
    ax_mem = fig.add_subplot(spec[:, 1])

    # Plot the data
    df["Time (s)"].plot(kind='bar', ax=ax_time)
    ax_time.set_ylim(ymin=0, ymax=ymax)
    title = f"dask/dask ({dask.__version__}) + zarr-developers/zarr-python ({zarr.__version__})" if plot_dask else "Zarr V3 Implementation"
    fig.legend(loc='outside upper center', ncol=LEGEND_COLS, title=title, borderaxespad=0)
    df["Memory (GB)"].plot(kind='bar', ax=ax_mem)

    # Styling
    ax_time.set_ylabel("Elapsed time (s)")
    ax_time.tick_params(axis='x', labelrotation=0)
    ax_time.grid(True, which='both', axis='y')
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)
    ax_mem.set_ylabel("Peak memory usage (GB)")
    ax_mem.tick_params(axis='x', labelrotation=0)
    ax_mem.grid(True, which='both', axis='y')
    ax_mem.spines['top'].set_visible(False)
    ax_mem.spines['right'].set_visible(False)

    custom_bar_label(ax_time)
    custom_bar_label(ax_mem)

    ax_time.get_legend().remove()
    ax_mem.get_legend().remove()

    fig.savefig(f"plots/benchmark_roundtrip{'_dask' if plot_dask else ''}.svg", metadata={'Date': None, 'Creator': None})
    # fig.savefig(f"plots/benchmark_roundtrip{'_dask' if plot_dask else ''}.pdf", metadata={'Date': None, 'Creator': None})


def main():
    plot_read_all(plot_dask=False, ymax=YMAX_READ_ALL)
    plot_read_all(plot_dask=True, ymax=YMAX_READ_ALL_DASK)
    plot_read_chunks(plot_dask=False)
    plot_read_chunks(plot_dask=True)
    plot_roundtrip(plot_dask=False, ymax=YMAX_ROUNDTRIP)
    plot_roundtrip(plot_dask=True, ymax=YMAX_ROUNDTRIP_DASK)
    plt.show()


if __name__ == "__main__":
    main()

