import pandas as pd
import numpy as np

def simulate_campaign(df, segment, budget, cost_per_action, conversion_rate, uplift_factor):
    """
    Simulates a marketing campaign on a specific segment.
    
    Args:
        df: DataFrame with predictions and segments
        segment: Target segment name
        budget: Total budget
        cost_per_action: Cost to target one customer
        conversion_rate: Probability of customer responding
        uplift_factor: Multiplier on their CLV if they respond (e.g. 1.1 for 10% increase)
        
    Returns:
        dict with campaign metrics
    """
    target_customers = df[df['segment_label'] == segment].copy()
    n_target = len(target_customers)
    
    if n_target == 0:
        return {
            'targeted_count': 0,
            'cost': 0,
            'revenue_uplift': 0,
            'roi': 0
        }
    
    # How many can we afford?
    max_affordable = int(budget / cost_per_action)
    n_targeted = min(n_target, max_affordable)
    
    # If we target top CLV within segment? Or random? 
    # Let's assume we target top predicted CLV first as optimal strategy
    target_customers = target_customers.sort_values('predicted_clv', ascending=False)
    selected = target_customers.iloc[:n_targeted]
    
    total_cost = n_targeted * cost_per_action
    
    # Expected Revenue without campaign
    base_revenue = selected['predicted_clv'].sum()
    
    # Expected Revenue with campaign
    # Uplift applies to those who convert
    # Expected Uplift = Base * (Uplift Factor - 1) * Conversion Rate
    # Note: This is simplified. Often uplift is structural.
    # Let's say: 
    # Responders get (CLV * uplift_factor)
    # Non-responders get (CLV)
    # E[Revenue] = (CLV * uplift * conv) + (CLV * (1-conv))
    #            = CLV * (uplift*conv + 1 - conv)
    #            = CLV * (1 + conv(uplift - 1))
    
    effective_multiplier = 1 + conversion_rate * (uplift_factor - 1)
    new_revenue = base_revenue * effective_multiplier
    
    revenue_uplift = new_revenue - base_revenue
    roi = (revenue_uplift - total_cost) / total_cost if total_cost > 0 else 0
    
    return {
        'total_in_segment': n_target,
        'targeted_count': n_targeted,
        'cost': total_cost,
        'base_revenue': base_revenue,
        'new_revenue': new_revenue,
        'revenue_uplift': revenue_uplift,
        'roi': roi
    }

def optimize_budget_allocation(df, budget, cost_per_action, conversion_rate, uplift_factor):
    """
    Optimizes budget allocation across all customers to maximize revenue uplift.
    
    Args:
        df: DataFrame with predictions
        budget: Total budget
        cost_per_action: Cost to target one customer
        conversion_rate: Probability of customer responding
        uplift_factor: Multiplier on their CLV if they respond
        
    Returns:
        dict with optimization results
    """
    # 1. Calculate Expected Lift Value per Customer
    # Lift = CLV * (Uplift Factor - 1) * Conversion Rate
    # This is the expected incremental revenue from targeting this specific customer
    
    # We use a copy to avoid modifying the session state df directly in a way that persists weirdly
    optim_df = df.copy()
    
    optim_df['expected_lift'] = optim_df['predicted_clv'] * (uplift_factor - 1) * conversion_rate
    
    # 2. Rank customers by Expected Lift (Greedy strategy)
    # We want to pick customers who give the most buck for the same cost (since cost is constant per action)
    optim_df = optim_df.sort_values('expected_lift', ascending=False)
    
    # 3. Select top N customers
    max_customers = int(budget / cost_per_action)
    selected_df = optim_df.iloc[:max_customers]
    
    # Metrics
    n_selected = len(selected_df)
    total_cost = n_selected * cost_per_action
    
    # Revenue calculations
    # Base revenue of selected set
    base_revenue_selected = selected_df['predicted_clv'].sum()
    
    # New revenue of selected set
    # New = Base * (1 + conv * (uplift - 1))
    effective_multiplier = 1 + conversion_rate * (uplift_factor - 1)
    new_revenue_selected = base_revenue_selected * effective_multiplier
    
    revenue_uplift = new_revenue_selected - base_revenue_selected
    roi = (revenue_uplift - total_cost) / total_cost if total_cost > 0 else 0
    
    # Compare against Random Selection (Baseline)
    # If we picked max_customers randomly
    if len(optim_df) > 0:
        random_df = optim_df.sample(n=min(len(optim_df), max_customers), random_state=42)
        random_base = random_df['predicted_clv'].sum()
        random_uplift = random_base * effective_multiplier - random_base
        random_roi = (random_uplift - total_cost) / total_cost if total_cost > 0 else 0
    else:
        random_uplift = 0
        random_roi = 0
        
    # Segment Allocation Breakdown
    allocation = selected_df['segment_label'].value_counts().reset_index()
    allocation.columns = ['Segment', 'Count']
    
    return {
        'targeted_count': n_selected,
        'cost': total_cost,
        'revenue_uplift': revenue_uplift,
        'roi': roi,
        'random_uplift': random_uplift,
        'random_roi': random_roi,
        'allocation': allocation,
        'selected_df': selected_df
    }

def calculate_efficient_frontier(df, max_budget, cost_per_action, conversion_rate, uplift_factor, steps=20):
    """
    Calculates the efficient frontier curve (Budget vs Uplift).
    """
    budgets = np.linspace(0, max_budget, steps)
    results = []
    
    for b in budgets:
        res = optimize_budget_allocation(df, b, cost_per_action, conversion_rate, uplift_factor)
        results.append({
            'Budget': b,
            'Revenue Uplift': res['revenue_uplift'],
            'ROI': res['roi']
        })
        
    return pd.DataFrame(results)

def perform_sensitivity_analysis(df, base_params, ranges=None):
    """
    Performs One-at-a-Time (OAT) sensitivity analysis on ROI.
    
    Args:
        df: Customer DataFrame
        base_params: dict with keys 'budget', 'cost_per_action', 'conversion_rate', 'uplift_factor'
        ranges: dict with keys matching base_params, containing lists/arrays of values to test.
                If None, defaults are used.
                
    Returns:
        dict of DataFrames, one for each parameter, containing 'Value' and 'ROI'.
    """
    
    # Defaults if not provided
    if ranges is None:
        ranges = {}
        
    # Define default ranges if missing
    if 'conversion_rate' not in ranges:
        ranges['conversion_rate'] = np.linspace(0.01, 0.30, 30)
    if 'uplift_factor' not in ranges:
        ranges['uplift_factor'] = np.linspace(1.0, 1.5, 20)
    if 'cost_per_action' not in ranges:
        # +/- 50% of base cost
        base_c = base_params['cost_per_action']
        if base_c == 0: base_c = 10 # Fallback
        ranges['cost_per_action'] = np.linspace(max(1, base_c * 0.5), base_c * 1.5, 20)
    if 'budget' not in ranges:
        base_b = base_params['budget']
        if base_b == 0: base_b = 5000 # Fallback
        ranges['budget'] = np.linspace(base_b * 0.5, base_b * 2.0, 20)
        
    results = {}
    
    # helper for simulation
    def get_roi(budget, cost, conv, uplift):
        # We use optimize_budget_allocation logic
        return optimize_budget_allocation(df, budget, cost, conv, uplift)['roi']

    # 1. Sensitivity: Conversion Rate
    res_conv = []
    for val in ranges['conversion_rate']:
        roi = get_roi(
            base_params['budget'], 
            base_params['cost_per_action'], 
            val, 
            base_params['uplift_factor']
        )
        res_conv.append({'Value': val, 'ROI': roi})
    results['conversion_rate'] = pd.DataFrame(res_conv)
    
    # 2. Sensitivity: Uplift Factor
    res_up = []
    for val in ranges['uplift_factor']:
        roi = get_roi(
            base_params['budget'], 
            base_params['cost_per_action'], 
            base_params['conversion_rate'], 
            val
        )
        res_up.append({'Value': val, 'ROI': roi})
    results['uplift_factor'] = pd.DataFrame(res_up)

    # 3. Sensitivity: Cost per Action
    res_cost = []
    for val in ranges['cost_per_action']:
        roi = get_roi(
            base_params['budget'], 
            val, 
            base_params['conversion_rate'], 
            base_params['uplift_factor']
        )
        res_cost.append({'Value': val, 'ROI': roi})
    results['cost_per_action'] = pd.DataFrame(res_cost)
    
    # 4. Sensitivity: Budget
    res_budg = []
    for val in ranges['budget']:
        roi = get_roi(
            val, 
            base_params['cost_per_action'], 
            base_params['conversion_rate'], 
            base_params['uplift_factor']
        )
        res_budg.append({'Value': val, 'ROI': roi})
    results['budget'] = pd.DataFrame(res_budg)
    
    return results

def perform_monte_carlo_simulation(df, base_params, std_devs, n_iterations=1000):
    """
    Performs Monte Carlo simulation for ROI uncertainty.
    
    Args:
        df: Customer DataFrame
        base_params: dict with keys 'budget', 'cost_per_action', 'conversion_rate', 'uplift_factor'
        std_devs: dict with keys 'conversion_rate', 'uplift_factor' (standard deviations)
        n_iterations: Number of simulation runs
        
    Returns:
        DataFrame with simulation results (Iteration, Conversion, Uplift, ROI, Revenue_Uplift)
    """
    
    # 1. Pre-calculate the base set of selected customers
    # The set of optimal customers depends on Rank of (CLV * Uplift * Conv).
    # Since Uplift and Conv are applied globally as scalars in this simulation model,
    # the RANKING of customers remains constant (checking if Uplift/Conv > 0).
    # Thus, we can select the top N customers once and re-use them.
    
    budget = base_params['budget']
    cost_per_action = base_params['cost_per_action']
    
    # Run one optimization to get the selected customers
    # We use base params for selection
    base_opt = optimize_budget_allocation(
        df, 
        budget, 
        cost_per_action, 
        base_params['conversion_rate'], 
        base_params['uplift_factor']
    )
    
    selected_df = base_opt['selected_df']
    n_selected = len(selected_df)
    total_cost = n_selected * cost_per_action
    
    if n_selected == 0:
        return pd.DataFrame()
        
    # Pre-sum the base CLV of selected customers
    base_revenue_selected = selected_df['predicted_clv'].sum()
    
    # 2. Generate Random Samples
    # Conversion Rate: Truncated Normal [0, 1]
    conv_mean = base_params['conversion_rate']
    conv_std = std_devs.get('conversion_rate', 0.05)
    
    # Uplift Factor: Truncated Normal [0, inf) (likely range 1.0 - 2.0)
    up_mean = base_params['uplift_factor']
    up_std = std_devs.get('uplift_factor', 0.1)
    
    # Use numpy for fast sampling
    # We sample simple normal then clip, which is "close enough" to truncated normal for small std devs
    # For rigorous stats we'd use scipy.stats.truncnorm, but this is fine for business estimates
    sim_conv = np.random.normal(conv_mean, conv_std, n_iterations)
    sim_conv = np.clip(sim_conv, 0.0, 1.0) # Clip to valid range
    
    sim_uplift = np.random.normal(up_mean, up_std, n_iterations)
    sim_uplift = np.clip(sim_uplift, 0.5, 5.0) # realistic bounds, uplift shouldn't be negative usually
    
    # 3. Vectorized Simulation
    # New Revenue = Base * (1 + conv * (uplift - 1))
    
    # effective_multiplier is an array of size n_iterations
    effective_multipliers = 1 + sim_conv * (sim_uplift - 1)
    
    # new_revenues is array
    new_revenues = base_revenue_selected * effective_multipliers
    
    revenue_uplifts = new_revenues - base_revenue_selected
    
    # ROI = (Uplift - Cost) / Cost
    # Cost is constant because we selected fixed N customers
    rois = (revenue_uplifts - total_cost) / total_cost if total_cost > 0 else np.zeros(n_iterations)
    
    # 4. Pack into DataFrame
    results_df = pd.DataFrame({
        'Iteration': range(n_iterations),
        'Conversion_Rate': sim_conv,
        'Uplift_Factor': sim_uplift,
        'Revenue_Uplift': revenue_uplifts,
        'ROI': rois
    })
    
    return results_df
