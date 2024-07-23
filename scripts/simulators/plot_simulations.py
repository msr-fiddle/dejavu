import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import sys
import pandas as pd
import argparse


red = 'tab:red'
blue = 'navy'
teal = 'teal'
black = 'k'
orange = 'orange'
lblue = 'lightsteelblue'
olive = 'tab:olive'
cyan = 'tab:cyan'
rblue = 'royalblue'
brown = 'brown'
green = 'green'

methods = ['Baseline', 'Baseline-DP', 'DejaVu']
colors = [rblue, green, red]

label_font_size = 34
width = 0.3
fig, ax = plt.subplots(figsize=(18, 8))


def get_data(csv_file):
    df = pd.read_csv(csv_file)

    baseline = list(df['Baseline'])
    baseline_dp = list(df['Baseline_dp'])
    dv = list(df['DV'])

    return baseline, baseline_dp, dv


def get_configs(csv_file):
    df = pd.read_csv(csv_file)

    baseline = list(df['Baseline_config'])
    baseline_dp = list(df['Baseline_dp_config'])
    dv = list(df['DV_config'])

    return baseline, baseline_dp, dv


def get_cost(csv_file):
    df = pd.read_csv(csv_file)

    baseline = list(df['Cost_baseline'])
    baseline_dp = list(df['Cost_baseline_dp'])
    dv = list(df['Cost_dv'])

    return baseline, baseline_dp, dv


def get_num_machines(csv_file):
    df = pd.read_csv(csv_file)
    return list(df['Num_machines'])


def plot_time(model, suffix):
    x = np.arange(len(data[0]))
    bars = []

    speedup_dp = [x/y for x, y in zip(data[0][1:], data[1][1:])]
    speedup_dv_baseline = [x/y for x, y in zip(data[0][1:], data[2][1:])]
    speedup_dv_dp = [x/y for x, y in zip(data[1][1:], data[2][1:])]

    print(f"SPEEDUP-DP: {max(speedup_dp)}")
    print(f"SPEEDUP-DV-BASELINE: {max(speedup_dv_baseline)}")
    print(f"SPEEDUP-DV-DP: {max(speedup_dv_dp)}")

    for i, method in enumerate(methods):
        bar = ax.bar(
            x + width * i, data[i], width,
            label=method,
            align='edge',
            color=colors[i]
        )
        bars.append(bar)

    x_tick_positions = x + width * len(methods) * 0.5
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=num_machines, fontsize=label_font_size
    )

    # ax.set_ylim(0,60000)
    plt.yticks(fontsize=label_font_size)
    # ax.set_ylim(0,300)
    ax.set_ylabel('Makespan (sec)', fontsize=label_font_size)
    ax.set_xlabel('Number of Machines', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper left', ncols=3,
               fontsize=label_font_size, bbox_to_anchor=(0.08, 1.18))

    # plt.show()
    plt.savefig(f"{model}_{suffix}.svg",
                bbox_inches="tight", dpi=500, pad_inches=0.1)


def plot_cost(model, suffix, machine_cost):
    x = np.arange(len(costs[0]))

    norm_costs = []
    for bs_cost in costs:
        norm_bs_costs = [x/machine_cost for x in bs_cost]
        norm_costs.append(norm_bs_costs)

    bars = []

    for i, method in enumerate(methods):
        bar = ax.bar(
            x + width * i, norm_costs[i], width,
            label=method,
            align='edge',
            color=colors[i]
        )
        bars.append(bar)

    x_tick_positions = x + width * len(methods) * 0.5
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=num_machines, fontsize=label_font_size
    )

    plt.yticks(fontsize=label_font_size)
    # ax.set_ylim(0,300)
    ax.set_ylabel('Normalized Cost', fontsize=label_font_size)
    ax.set_xlabel('Number of Machines', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper left', ncols=3,
               fontsize=label_font_size, bbox_to_anchor=(0.08, 1.18))

    # plt.show()
    plt.savefig(f"{model}_{suffix}_cost.svg",
                bbox_inches="tight", dpi=500, pad_inches=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_cost', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--machine_cost', type=float)

    args = parser.parse_args()

    data = get_data(args.input)
    costs = get_cost(args.input)
    num_machines = get_num_machines(args.input)

    print(data)
    print(costs)

    if args.plot_cost:
        plot_cost(args.model, args.suffix, args.machine_cost)
    else:
        plot_time(args.model, args.suffix)
