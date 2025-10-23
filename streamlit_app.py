# Harris & Bullock (2002) coevolutionary signalling model – full Python version
# Includes both signaller and receiver populations, strategy classification, and positive-selection heatmaps.

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# === Easily adjustable parameters ===
pop_size     = 100       # per population
generations  = 1000      # evolutionary generations
replicates   = 20        # number of replicate runs per (CH,CL) combo
mutation_rate = 0.03
tournament_size = 2
levels = list(range(-200, 201, 50))   # CH, CL grid: -200..200 step 50 (9 levels)
# ====================================

# --- Strategy name helpers ---
def signaller_strategy_name(gen):
    b0, b1 = gen[0], gen[1]
    if b0==0 and b1==0: return "Cynic"
    if b0==1 and b1==1: return "Bluffer"
    if b0==0 and b1==1: return "Honest"
    if b0==1 and b1==0: return "Dishonest"

def receiver_strategy_name(gen):
    bE, bW = gen[0], gen[1]
    if bE==0 and bW==0: return "Mean"
    if bE==1 and bW==1: return "Generous"
    if bE==0 and bW==1: return "Believer"
    if bE==1 and bW==0: return "Non-believer"

def receiver_response(gen, signal):
    # signal: 0=East, 1=West
    return gen[signal]

# --- Game interaction ---
def play_generation(signallers, receivers, c_EH, c_WH, c_EL, c_WL):
    N = len(signallers)
    states = np.random.choice([0,1], size=N)  # 0=Low, 1=High
    rec_idx = np.random.permutation(N)
    fit_S = np.zeros(N)
    fit_R = np.zeros(N)
    for i in range(N):
        s = signallers[i]
        r = receivers[rec_idx[i]]
        state = states[i]
        action = s[state]           # signal chosen by signaller
        response = receiver_response(r, action)
        # receiver payoff
        if (response==1 and state==1) or (response==0 and state==0):
            fit_R[rec_idx[i]] += 100
        # signaller payoff: gets 100 if Up response minus cost
        if action==0 and state==1: cost = c_EH
        elif action==1 and state==1: cost = c_WH
        elif action==0 and state==0: cost = c_EL
        else: cost = c_WL
        value = 100 if response==1 else 0
        fit_S[i] += (value - cost)
    return fit_S, fit_R

# --- Evolutionary reproduction ---
def reproduce_tournament(pop, fitness, mutation_rate):
    N = len(pop)
    new_pop = np.zeros_like(pop)
    for i in range(N):
        inds = np.random.choice(N, size=tournament_size, replace=False)
        winner = inds[0]
        if fitness[inds[1]] > fitness[winner]:
            winner = inds[1]
        child = pop[winner].copy()
        # mutation
        for g in range(len(child)):
            if random.random() < mutation_rate:
                child[g] = 1 - child[g]
        new_pop[i] = child
    return new_pop

# --- Run one replicate ---
def run_replicate(CH, CL, generations=generations):
    # cost setup (simplified 2D test)
    c_EH = 0; c_EL = 0; c_WH = CH; c_WL = CL
    # initialise populations
    signallers = np.random.choice([0,1], size=(pop_size,2))
    receivers  = np.random.choice([0,1], size=(pop_size,2))
    # evolve
    for gen in range(generations):
        fit_S, fit_R = play_generation(signallers, receivers, c_EH, c_WH, c_EL, c_WL)
        signallers = reproduce_tournament(signallers, fit_S, mutation_rate)
        receivers  = reproduce_tournament(receivers,  fit_R, mutation_rate)
    # final counts
    s_counts = {k:0 for k in ["Cynic","Bluffer","Honest","Dishonest"]}
    r_counts = {k:0 for k in ["Mean","Generous","Believer","Non-believer"]}
    for g in signallers: s_counts[signaller_strategy_name(g)] += 1
    for g in receivers:  r_counts[receiver_strategy_name(g)] += 1
    return s_counts, r_counts

# --- Run grid of parameter combinations ---
def run_grid():
    n_levels = len(levels)
    # storage
    s_pos = {k: np.zeros((n_levels, n_levels), dtype=int) for k in ["Cynic","Bluffer","Honest","Dishonest"]}
    r_pos = {k: np.zeros((n_levels, n_levels), dtype=int) for k in ["Mean","Generous","Believer","Non-believer"]}

    total_runs = len(levels) * len(levels) * replicates
    print(f"\nRunning {len(levels)}×{len(levels)} parameter grid with {replicates} replicates each")
    print(f"Total simulations: {total_runs}\n")
    
    with tqdm(total=total_runs, desc="Overall progress", unit="sim") as pbar:
        for i, CH in enumerate(levels):
            for j, CL in enumerate(levels):
                for rep in range(replicates):
                    s_counts, r_counts = run_replicate(CH, CL)
                    for strat in s_counts:
                        if s_counts[strat] > 37:
                            s_pos[strat][j,i] += 1
                    for strat in r_counts:
                        if r_counts[strat] > 37:
                            r_pos[strat][j,i] += 1
                    pbar.update(1)
                    pbar.set_postfix({"CH": CH, "CL": CL, "rep": f"{rep+1}/{replicates}"})
    
    print("\n✓ Simulation complete!\n")
    return s_pos, r_pos

# --- Visualization helper ---
def plot_strategy_heatmaps(pos_counts, title_prefix):
    for strat, grid in pos_counts.items():
        fig, ax = plt.subplots(figsize=(6,5))
        
        # Calculate bin edges (boundaries between cells)
        step = levels[1] - levels[0] if len(levels) > 1 else 50
        extent = [min(levels) - step/2, max(levels) + step/2, 
                  min(levels) - step/2, max(levels) + step/2]
        
        im = ax.imshow(grid, origin='lower', aspect='auto',
                       extent=extent, cmap='viridis')
        plt.colorbar(im, ax=ax, label=f"# replicates w/ positive selection (out of {replicates})")
        
        # Set ticks at the actual level values
        ax.set_xticks(levels)
        ax.set_yticks(levels)
        
        ax.set_xlabel("CH (c(W,H))")
        ax.set_ylabel("CL (c(W,L))")
        ax.set_title(f"{title_prefix}: {strat}")
        plt.tight_layout()
        plt.show()

# === Run simulation grid ===
if __name__ == "__main__":
    print("=" * 60)
    print("Harris & Bullock Coevolutionary Signalling Model")
    print("=" * 60)
    print(f"Population size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Replicates per condition: {replicates}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"Cost levels: {levels}")
    
    s_positive, r_positive = run_grid()

    # === Plot signaller results ===
    print("Generating signaller strategy heatmaps...")
    plot_strategy_heatmaps(s_positive, "Signaller strategy positive selection")

    # === Plot receiver results ===
    print("Generating receiver strategy heatmaps...")
    plot_strategy_heatmaps(r_positive, "Receiver strategy positive selection")

    # === Example: inspect dataframes ===
    df_receivers = pd.DataFrame(r_positive["Believer"], index=levels, columns=levels)
    print("\nBeliever strategy positive-selection counts:\n")
    print(df_receivers)

    # === Save results to CSV ===
    print("\nSaving results to CSV files...")
    for strat, grid in s_positive.items():
        pd.DataFrame(grid, index=levels, columns=levels).to_csv(f"signaller_{strat}.csv")
    
    for strat, grid in r_positive.items():
        pd.DataFrame(grid, index=levels, columns=levels).to_csv(f"receiver_{strat}.csv")
    
    print("✓ All files saved successfully!")
