import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import zipfile

# Set page config
st.set_page_config(
    page_title="Harris & Bullock Coevolution Model",
    page_icon="🧬",
    layout="wide"
)

# Title and description
st.title("🧬 Harris & Bullock (2002) Coevolutionary Signalling Model")
st.markdown("""
This simulation models the coevolution of signallers and receivers in a communication game.
Adjust parameters in the sidebar and run the simulation to see which strategies evolve under different cost conditions.
""")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")

pop_size = st.sidebar.number_input("Population Size", min_value=10, max_value=500, value=100, step=10)
generations = st.sidebar.number_input("Generations", min_value=100, max_value=5000, value=1000, step=100)
replicates = st.sidebar.number_input("Replicates per condition", min_value=1, max_value=50, value=20, step=1)
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.1, value=0.03, step=0.01)
tournament_size = st.sidebar.selectbox("Tournament Size", [2, 3, 4, 5], index=0)

st.sidebar.header("Cost Grid Parameters")
min_cost = st.sidebar.number_input("Minimum Cost", min_value=-500, max_value=0, value=-200, step=50)
max_cost = st.sidebar.number_input("Maximum Cost", min_value=0, max_value=500, value=200, step=50)
step_size = st.sidebar.number_input("Step Size", min_value=10, max_value=100, value=50, step=10)

levels = list(range(min_cost, max_cost + 1, step_size))

# Display summary
st.sidebar.markdown("---")
st.sidebar.markdown("**Summary**")
st.sidebar.info(f"""
Grid size: {len(levels)} × {len(levels)}  
Total simulations: {len(levels) * len(levels) * replicates}  
Estimated time: ~{(len(levels) * len(levels) * replicates * generations) // 10000} seconds
""")

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
    return gen[signal]

# --- Game interaction ---
def play_generation(signallers, receivers, c_EH, c_WH, c_EL, c_WL):
    N = len(signallers)
    states = np.random.choice([0,1], size=N)
    rec_idx = np.random.permutation(N)
    fit_S = np.zeros(N)
    fit_R = np.zeros(N)
    for i in range(N):
        s = signallers[i]
        r = receivers[rec_idx[i]]
        state = states[i]
        action = s[state]
        response = receiver_response(r, action)
        if (response==1 and state==1) or (response==0 and state==0):
            fit_R[rec_idx[i]] += 100
        if action==0 and state==1: cost = c_EH
        elif action==1 and state==1: cost = c_WH
        elif action==0 and state==0: cost = c_EL
        else: cost = c_WL
        value = 100 if response==1 else 0
        fit_S[i] += (value - cost)
    return fit_S, fit_R

# --- Evolutionary reproduction ---
def reproduce_tournament(pop, fitness, mutation_rate, tournament_size):
    N = len(pop)
    new_pop = np.zeros_like(pop)
    for i in range(N):
        inds = np.random.choice(N, size=tournament_size, replace=False)
        winner = inds[np.argmax(fitness[inds])]
        child = pop[winner].copy()
        for g in range(len(child)):
            if random.random() < mutation_rate:
                child[g] = 1 - child[g]
        new_pop[i] = child
    return new_pop

# --- Run one replicate ---
def run_replicate(CH, CL, pop_size, generations, mutation_rate, tournament_size):
    c_EH = 0; c_EL = 0; c_WH = CH; c_WL = CL
    signallers = np.random.choice([0,1], size=(pop_size,2))
    receivers  = np.random.choice([0,1], size=(pop_size,2))
    for gen in range(generations):
        fit_S, fit_R = play_generation(signallers, receivers, c_EH, c_WH, c_EL, c_WL)
        signallers = reproduce_tournament(signallers, fit_S, mutation_rate, tournament_size)
        receivers  = reproduce_tournament(receivers,  fit_R, mutation_rate, tournament_size)
    s_counts = {k:0 for k in ["Cynic","Bluffer","Honest","Dishonest"]}
    r_counts = {k:0 for k in ["Mean","Generous","Believer","Non-believer"]}
    for g in signallers: s_counts[signaller_strategy_name(g)] += 1
    for g in receivers:  r_counts[receiver_strategy_name(g)] += 1
    return s_counts, r_counts

# --- Run grid of parameter combinations ---
def run_grid(levels, pop_size, generations, replicates, mutation_rate, tournament_size, progress_bar, status_text):
    n_levels = len(levels)
    s_pos = {k: np.zeros((n_levels, n_levels), dtype=int) for k in ["Cynic","Bluffer","Honest","Dishonest"]}
    r_pos = {k: np.zeros((n_levels, n_levels), dtype=int) for k in ["Mean","Generous","Believer","Non-believer"]}

    total_runs = len(levels) * len(levels) * replicates
    current = 0
    
    for i, CH in enumerate(levels):
        for j, CL in enumerate(levels):
            for rep in range(replicates):
                s_counts, r_counts = run_replicate(CH, CL, pop_size, generations, mutation_rate, tournament_size)
                for strat in s_counts:
                    if s_counts[strat] > 37:
                        s_pos[strat][j,i] += 1
                for strat in r_counts:
                    if r_counts[strat] > 37:
                        r_pos[strat][j,i] += 1
                current += 1
                progress_bar.progress(current / total_runs)
                status_text.text(f"CH={CH}, CL={CL}, replicate {rep+1}/{replicates} ({current}/{total_runs})")
    
    return s_pos, r_pos

# --- Visualization helper ---
def create_heatmap(grid, strat, levels, replicates):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate bin edges (boundaries between cells)
    step = levels[1] - levels[0] if len(levels) > 1 else 50
    extent = [min(levels) - step/2, max(levels) + step/2, 
              min(levels) - step/2, max(levels) + step/2]
    
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=extent, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax, label=f"# replicates w/ positive selection (out of {replicates})")
    
    # Set ticks at the actual level values
    ax.set_xticks(levels)
    ax.set_yticks(levels)
    
    ax.set_xlabel("CH (c(W,H))", fontsize=12)
    ax.set_ylabel("CL (c(W,L))", fontsize=12)
    ax.set_title(f"{strat}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# Main app
if st.button("🚀 Run Simulation", type="primary"):
    st.markdown("---")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run simulation
    with st.spinner("Running simulation..."):
        s_positive, r_positive = run_grid(levels, pop_size, generations, replicates, 
                                          mutation_rate, tournament_size, progress_bar, status_text)
    
    status_text.text("✅ Simulation complete!")
    
    # Display results
    st.markdown("---")
    st.header("📊 Results")
    
    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["Signaller Strategies", "Receiver Strategies", "Data Export"])
    
    with tab1:
        st.subheader("Signaller Strategy Positive Selection")
        cols = st.columns(2)
        for idx, (strat, grid) in enumerate(s_positive.items()):
            with cols[idx % 2]:
                fig = create_heatmap(grid, strat, levels, replicates)
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        st.subheader("Receiver Strategy Positive Selection")
        cols = st.columns(2)
        for idx, (strat, grid) in enumerate(r_positive.items()):
            with cols[idx % 2]:
                fig = create_heatmap(grid, strat, levels, replicates)
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        st.subheader("Export Data")
        
        # Create downloadable CSV files
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for strat, grid in s_positive.items():
                df = pd.DataFrame(grid, index=levels, columns=levels)
                csv_buffer = BytesIO()
                df.to_csv(csv_buffer)
                zip_file.writestr(f"signaller_{strat}.csv", csv_buffer.getvalue())
            
            for strat, grid in r_positive.items():
                df = pd.DataFrame(grid, index=levels, columns=levels)
                csv_buffer = BytesIO()
                df.to_csv(csv_buffer)
                zip_file.writestr(f"receiver_{strat}.csv", csv_buffer.getvalue())
        
        st.download_button(
            label="📥 Download All Results (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="coevolution_results.zip",
            mime="application/zip"
        )
        
        # Show sample data
        st.markdown("**Sample: Believer Strategy**")
        df_believer = pd.DataFrame(r_positive["Believer"], index=levels, columns=levels)
        st.dataframe(df_believer)

else:
    st.info("👈 Adjust parameters in the sidebar and click 'Run Simulation' to start")
    
    # Show strategy descriptions
    st.markdown("---")
    st.header("📖 Strategy Descriptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Signaller Strategies")
        st.markdown("""
        - **Cynic**: Never signals (0,0)
        - **Bluffer**: Always signals (1,1)
        - **Honest**: Signals only in high state (0,1)
        - **Dishonest**: Signals only in low state (1,0)
        """)
    
    with col2:
        st.subheader("Receiver Strategies")
        st.markdown("""
        - **Mean**: Never responds positively (0,0)
        - **Generous**: Always responds positively (1,1)
        - **Believer**: Responds to West signal (0,1)
        - **Non-believer**: Responds to East signal (1,0)
        """)
