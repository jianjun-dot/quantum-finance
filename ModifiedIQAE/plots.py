import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import sys
from collections import defaultdict

def pad_matrix(arr):
    max_len = len(max(arr, key=lambda x: len(x)))

    padded = np.zeros((len(arr), max_len), dtype=int)

    for i in range(len(arr)):
        padded[i:i + 1, 0:len(arr[i])] = arr[i]
        
    return padded

def make_plots(experiment_dir: str, dest_dir=None):
    
    if not dest_dir:
        dest_dir = os.path.join(experiment_dir, 'results_images')
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
    
    # load per-round files
    per_round_path = os.path.join(experiment_dir, 'per_round')
    per_round_stats = []

    for csv_name in os.listdir(per_round_path):
        csv_path = os.path.join(per_round_path, csv_name)
        confint_method, epsilon, alg = csv_name.split('_')[1:4]

        with open(csv_path) as f:
            rows = csv.reader(f)
            for csv_row in rows:
                df_row = [confint_method, epsilon, alg, csv_row[0], np.array([float(x) for x in csv_row[1:]])]
                per_round_stats.append(df_row)

    per_round_df = pd.DataFrame(per_round_stats, columns=['confint_method', 'epsilon', 'alg', 'field', 'data'])
    fields = per_round_df['field'].unique()

    per_round_df.loc[per_round_df['alg'] == 'miae', 'alg'] = 'miqae'
    per_round_df.loc[per_round_df['alg'] == 'iae', 'alg'] = 'iqae'

    names = ['shots_vs_ki', 'queries_vs_ki', 'ki_vs_i']

    # per-round
    plt.rcParams.update({'font.size': 20})

    for epsilon, df_epsilon in per_round_df.groupby(['epsilon']):

        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        figs = [plt.figure(figsize=(5,5)) for i in range(3)]
        for fig in figs: fig.add_axes([0,0,1,1])
        axs = [fig.axes[0] for fig in figs]

        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('k_i')
        axs[0].set_ylabel('Shots per round')

        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('k_i')
        axs[1].set_ylabel('Queries per round')

        axs[2].set_yscale('log')
        axs[2].set_xlabel('Round number')
        axs[2].set_ylabel('k_i')

        n_entries = 0
        
        for (alg, confint_method), df_alg_confint in sorted(df_epsilon.groupby(['alg', 'confint_method']), 
                                                            key=lambda x: (x[0][0], -ord(x[0][1][0]))):
        
            
            avg_stats = {}

            for field in fields:
                padded_stats = pad_matrix(df_alg_confint[df_alg_confint['field'] == field]['data'].to_numpy().tolist())
                div = np.count_nonzero(padded_stats, axis=0)
                div[div == 0] = 1 # suppress divide by zero
                avg_stats[field] = padded_stats.sum(axis=0) / div
                avg_stats[field][avg_stats[field] == 0] = 0.1 # for adding zero to log plots
                
            n_entries = max(n_entries, len(avg_stats['round_k']))
            label = f'{alg.upper()}+{confint_method.title()}'

            # plots for shots vs k
            axs[0].plot(avg_stats['round_k'], avg_stats['round_shots'], label=label)
            axs[0].scatter(avg_stats['round_k'], avg_stats['round_shots'])

            # plots for nqueries vs k
            axs[1].plot(avg_stats['round_k'], avg_stats['round_queries'], label=label)
            axs[1].scatter(avg_stats['round_k'], avg_stats['round_queries'])

            # plots for k
            axs[2].plot(avg_stats['round_k'], label=label)
            axs[2].scatter(range(len(avg_stats['round_k'])), avg_stats['round_k'])

        for i in range(3): axs[i].legend()

        axs[2].plot(range(-10, n_entries+5), np.repeat(np.pi / 4 / float(epsilon), n_entries+15), c='r', linestyle='--')
        axs[2].set_xlim(-2, n_entries+2)

        for i in range(3):
            figs[i].savefig(os.path.join(dest_dir, f'{epsilon}_per_round_{names[i]}.png'), bbox_inches='tight')
        plt.close()
            
    # overall query complexity and failure rate
    df_rows = []

    for csv_name in os.listdir(experiment_dir):
        csv_path = os.path.join(experiment_dir, csv_name)
        if os.path.splitext(csv_name)[1] != '.csv': continue
        amplitude, confint_method, alg = csv_name.split('_')[:3]

        with open(csv_path) as f:
            rows = csv.reader(f)
            for csv_row in rows:
                epsilon, failure, data = float(csv_row[0]), float(csv_row[1]), np.array([int(x) for x in csv_row[2:]]).mean()
                df_rows.append([alg, confint_method, epsilon, amplitude, data, failure])


    df = pd.DataFrame(df_rows, columns=['alg', 'confint_method', 'epsilon', 'amplitude', 'data', 'failure'])

    df.loc[df['alg'] == 'miae', 'alg'] = 'miqae'
    df.loc[df['alg'] == 'iae', 'alg'] = 'iqae'

    complexity = defaultdict(lambda: defaultdict(list))

    # plt.rcParams.update({'font.size': 14})

    for (alg, confint_method, epsilon), df_i in sorted(df.groupby(['alg', 'confint_method', 'epsilon']), 
                                                       key=lambda x: (-x[0][2], x[0][0], -ord(x[0][1][0]))):
        complexity[(alg,confint_method)]['epsilon'].append(epsilon)
        complexity[(alg,confint_method)]['queries'].append(df_i['data'].mean())

    plt.figure(figsize=(15,7))
    plt.xscale('log')
    plt.yscale('log')

    epsilons = sorted(df['epsilon'].unique(), reverse=True)

    plt.xlim(epsilons[0]*2, epsilons[-1]/2)
    plt.xlabel('Epsilon')
    plt.ylabel('Number of queries')

    for (alg, confint_method), data in complexity.items():
        epsilon, queries = data['epsilon'], data['queries']
        plt.scatter(epsilon, queries)
        plt.plot(epsilon, queries, label=f'{alg.upper()}+{confint_method.title()}')

    plt.legend()

    plt.savefig(os.path.join(dest_dir, 'complexity.png'), bbox_inches='tight')
    plt.close()

    # Query count vs. input amplitude

    amplitude_query = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )

    for (alg, confint_method, amplitude, epsilon), df_i in sorted(df.groupby(['alg', 'confint_method', 'amplitude', 'epsilon']), 
                                                                key=lambda x: (x[0][2], x[0][0], -ord(x[0][1][0]))):
        
        amplitude_query[alg][confint_method][epsilon]['amplitudes'].append(amplitude)
        amplitude_query[alg][confint_method][epsilon]['queries'].append(df_i['data'].mean())

    plt.rcParams.update({'font.size': 14})

    for alg in ['miqae', 'iqae']:
        plt.figure()
        plt.yscale('log')
        
        for i,confint_method in enumerate(['chernoff', 'beta']):
            for epsilon, color in zip(amplitude_query[alg][confint_method], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']):
                amplitudes = amplitude_query[alg][confint_method][epsilon]['amplitudes']
                queries = amplitude_query[alg][confint_method][epsilon]['queries']
                plt.plot(amplitudes, queries, color=color, linestyle='dashed' if confint_method == 'beta' else 'solid')
                plt.scatter(amplitudes, queries, color=color, 
                            label='{:.0e}'.format(epsilon),
                            marker='^' if confint_method == 'beta' else '.')
                
            if i == 0:
                plt.legend()
                
        # plt.title(f'Number of queries vs. Input amplitude, {alg.upper()}')
        plt.xlabel('Input amplitude')
        plt.ylabel('Number of queries')
        
        locs,labels = plt.xticks()
        plt.xticks(range(0, int(locs[-1])+1, 4)) # make more flexible

        plt.savefig(os.path.join(dest_dir, f'queries-vs-amplitude_{alg}.png'), bbox_inches='tight')
        plt.close()


    # Failure rate

    for amplitude, df_i in sorted(df.groupby('amplitude')):

        plt.figure()
        x = np.arange(len(epsilons))  # the label locations
        width = 0.2 # the width of the bars

        chernoff_miae_failures = df_i[(df_i['alg'] == 'miqae') & (df_i['confint_method'] == 'chernoff')]['failure']
        chernoff_iae_failures = df_i[(df_i['alg'] == 'iqae') & (df_i['confint_method'] == 'chernoff')]['failure']
        beta_miae_failures = df_i[(df_i['alg'] == 'miqae') & (df_i['confint_method'] == 'beta')]['failure']
        beta_iae_failures = df_i[(df_i['alg'] == 'iqae') & (df_i['confint_method'] == 'beta')]['failure']

        rects1 = plt.bar(x - 3*width/2, chernoff_iae_failures, width, label='IQAE+Chernoff')
        rects2 = plt.bar(x - width/2, beta_iae_failures, width, label='IQAE+Beta')
        rects3 = plt.bar(x + width/2, chernoff_miae_failures, width, label='MIQAE+Chernoff')
        rects4 = plt.bar(x + 3*width/2, beta_miae_failures, width, label='MIQAE+Beta')

        plt.ylabel('Failure ratio')
        plt.xlabel('Epsilon')
        plt.xticks(x, epsilons)
        plt.ylim(0,0.06)
    
        line_xs = np.arange(len(epsilons)+3) - 2

        plt.plot(line_xs, [0.05] * len(line_xs), linestyle='--', c='r')
        plt.xlim(-0.5, len(epsilons)-.5)
        
        plt.legend()

        plt.savefig(os.path.join(dest_dir, f'failures_{amplitude}.png'), bbox_inches='tight')
        # plt.show()
        plt.close()


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('No result path was provided.')
    else:
        results_path = sys.argv[1]
        make_plots(results_path)
        