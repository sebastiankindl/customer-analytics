import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from segmentation.segment import assign_segments
from business.roi import simulate_campaign, optimize_budget_allocation, calculate_efficient_frontier, perform_sensitivity_analysis, perform_monte_carlo_simulation

# Page Config
st.set_page_config(page_title="CLV Prediction Engine", layout="wide")

# --- Load Data & Model ---
@st.cache_resource
def load_resources():
    # Load Model
    with open('models/clv_xgb.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load Data (Test set acts as "Current Customers" for demo)
    df = pd.read_csv('data/processed/test.csv')
    return model, df

@st.cache_resource
def load_shap():
    try:
        with open('models/shap_values.pkl', 'rb') as f:
            shap_values = pickle.load(f)
        return shap_values
    except FileNotFoundError:
        return None

try:
    model, df = load_resources()
    shap_values = load_shap()
except FileNotFoundError:
    st.error("Model or Data not found. Please run the pipeline first.")
    st.stop()

# --- Predictions & Processing ---
if 'predictions_done' not in st.session_state:
    features = ['recency', 'frequency', 'monetary', 'tenure', 'aov', 'avg_inter_purchase_time']
    X = df[features]
    
    # Predict
    preds = model.predict(X)
    df['predicted_clv'] = np.maximum(preds, 0)
    
    # Segment
    df = assign_segments(df)
    
    st.session_state['df'] = df
    st.session_state['predictions_done'] = True
else:
    df = st.session_state['df']
    features = ['recency', 'frequency', 'monetary', 'tenure', 'aov', 'avg_inter_purchase_time']

# --- Sidebar ---
st.sidebar.title("CLV Engine")
page = st.sidebar.radio("Navigation", ["Overview", "Drivers of Value", "Customer Explorer", "ROI Simulator"])

# --- Overview Page ---
if page == "Overview":
    st.title("Est. Future Revenue Overview (Next 90 Days)")
    
    col1, col2, col3 = st.columns(3)
    
    total_pred_rev = df['predicted_clv'].sum()
    avg_pred_clv = df['predicted_clv'].mean()
    high_value_count = df[df['value_segment'] == 'High Value'].shape[0]
    
    col1.metric("Total Predicted Revenue", f"${total_pred_rev:,.0f}")
    col2.metric("Avg CLV per Customer", f"${avg_pred_clv:.2f}")
    col3.metric("High Value Customers", f"{high_value_count}")
    
    # --- Model Performance Panel ---
    st.markdown("---")
    st.subheader("Model Performance")
    
    try:
        with open('reports/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        
        xgb_m = metrics['xgboost']
        base_m = metrics['baseline']
        
        # Helper to format delta
        def fmt_delta(curr, base, inverse=True):
            delta = (curr - base) / base
            if inverse: delta = -delta # Lower is better (MAE/RMSE)
            return f"{delta:.1%}"

        m1.metric("MAE (Error)", f"${xgb_m['MAE']:.2f}", fmt_delta(xgb_m['MAE'], base_m['MAE']))
        m2.metric("RMSE", f"${xgb_m['RMSE']:.2f}", fmt_delta(xgb_m['RMSE'], base_m['RMSE']))
        m3.metric("R² Score", f"{xgb_m['R2']:.3f}", fmt_delta(xgb_m['R2'], base_m['R2'], inverse=False))
        m4.metric("Top Decile Lift", f"{xgb_m['lift']:.2f}x", fmt_delta(xgb_m['lift'], base_m['lift'], inverse=False))
        
        # Visual Comparison
        p1, p2 = st.columns(2)
        
        with p1:
            # Bar Chart: MAE Comparison
            perf_df = pd.DataFrame({
                'Model': ['Baseline', 'XGBoost'],
                'MAE': [base_m['MAE'], xgb_m['MAE']],
                'RMSE': [base_m['RMSE'], xgb_m['RMSE']]
            })
            
            fig_perf = px.bar(perf_df, x='Model', y=['MAE', 'RMSE'], barmode='group', 
                              title="Model Error Comparison (Lower is Better)")
            st.plotly_chart(fig_perf, use_container_width=True)
            
        with p2:
            # Capture Rate Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = xgb_m['capture_rate'] * 100,
                title = {'text': "Top Decile Revenue Capture %"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgray"}, # Random baseline
                        {'range': [10, 50], 'color': "lightblue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 10 # Random baseline is 10%
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
    except FileNotFoundError:
        st.warning("Metrics file not found. Run training script.")
    
    st.markdown("---")
    
    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Revenue Distribution")
        fig_hist = px.histogram(df, x='predicted_clv', nbins=50, title="Predicted CLV Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        st.subheader("Segments")
        fig_seg = px.scatter(df, x='recency', y='predicted_clv', color='segment_label', 
                             title="CLV vs Recency by Segment", hover_data=['customer_id'])
        st.plotly_chart(fig_seg, use_container_width=True)


# --- Drivers of Value (Explainability) ---
elif page == "Drivers of Value":
    st.title("Drivers of Customer Value")
    
    st.markdown("""
    Understanding **why** the model predicts high customer value is critical for strategy.
    This page shows which features drive the predictions globally.
    """)
    
    if os.path.exists('reports/shap_importance.csv'):
        imp_df = pd.read_csv('reports/shap_importance.csv')
        
        # Plot Global Importance
        fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h', 
                         title="Global Feature Importance (SHAP)",
                         labels={'importance': 'Mean Absolute SHAP Value (Impact on Prediction)', 'feature': 'Feature'})
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Business Interpretation
        st.subheader("Business Interpretation")
        top_feature = imp_df.iloc[0]['feature']
        st.info(f"The most important driver of CLV is **{top_feature}**. Improving this metric typically yields the highest revenue uplift.")
        
        st.markdown("""
        - **Monetary**: Total historical spend. High past spend strongly predicts high future spend.
        - **Avg Inter-Purchase Time**: Consistency matters. Regular buyers are more predictable.
        - **Recency**: Recent buyers are more likely to buy again soon.
        - **Frequency**: More transactions generally indicate higher engagement.
        """)
        
    else:
        st.warning("SHAP importance not found. Please train the model.")

# --- Customer Explorer ---
elif page == "Customer Explorer":
    st.title("Customer Explorer")
    
    col_filter, col_select = st.columns([2, 1])
    
    with col_filter:
        segment_filter = st.multiselect("Filter by Segment", options=df['segment_label'].unique(), default=df['segment_label'].unique())
    
    filtered_df = df[df['segment_label'].isin(segment_filter)].sort_values('predicted_clv', ascending=False)
    
    with col_select:
        # Select specific customer
        selected_customer_id = st.selectbox("Select Customer to Inspect", filtered_df['customer_id'].unique())
    
    # 1. Show Dataframe
    st.dataframe(filtered_df[['customer_id', 'recency', 'frequency', 'monetary', 'predicted_clv', 'segment_label', 'recommended_action']],
                 use_container_width=True, height=250)
    
    # 2. Individual Customer Explanation
    if selected_customer_id and shap_values is not None:
        st.markdown("---")
        st.subheader(f"Why is Customer {selected_customer_id} predicted this value?")
        
        # Get index of customer in the test set (filtered_df might be subset, need original DF index?)
        # NOTE: shap_values corresponds to 'df' (test set) row order if loaded correctly.
        # Ensure 'df' index is aligned with shap_values indices [0..N]
        
        # Find integer index of the customer in the full 'df'
        cust_idx = df[df['customer_id'] == selected_customer_id].index[0]
        
        # SHAP values for this customer
        cust_shap = shap_values[cust_idx]
        base_value = cust_shap.base_values if hasattr(cust_shap, 'base_values') else cust_shap.mean() # Hack if raw array
        # TreeExplainer usually returns array for values
        
        # Safely handle shap_values structure (it can be list or array depending on version/model)
        if isinstance(shap_values, list):
             cust_vals = shap_values[0][cust_idx]
        elif hasattr(shap_values, 'values'): # Explain object
             cust_vals = shap_values.values[cust_idx]
        else:
             cust_vals = shap_values[cust_idx]

        # Create Waterfall-like data
        # Feature | Value | Impact
        expl_df = pd.DataFrame({
            'Feature': features,
            'Feature Value': df.iloc[cust_idx][features].values,
            'Impact (SHAP)': cust_vals
        })
        
        expl_df['Color'] = np.where(expl_df['Impact (SHAP)'] > 0, 'Positive (Increases Value)', 'Negative (Decreases Value)')
        expl_df = expl_df.sort_values('Impact (SHAP)', key=abs, ascending=True)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.write(f"**Predicted CLV:** ${df.iloc[cust_idx]['predicted_clv']:.2f}")
            st.metric("Prediction", f"${df.iloc[cust_idx]['predicted_clv']:.2f}")
            
            # Simple Bar Chart for Impact
            fig_explain = px.bar(expl_df, x='Impact (SHAP)', y='Feature', color='Color', orientation='h',
                                 title="Individual Prediction Drivers",
                                 color_discrete_map={'Positive (Increases Value)': 'green', 'Negative (Decreases Value)': 'red'})
            st.plotly_chart(fig_explain, use_container_width=True)
            
        with c2:
            st.write(" **Interpretation:**")
            top_pos = expl_df[expl_df['Impact (SHAP)'] > 0].sort_values('Impact (SHAP)', ascending=False).head(1)
            top_neg = expl_df[expl_df['Impact (SHAP)'] < 0].sort_values('Impact (SHAP)', ascending=True).head(1)
            
            if not top_pos.empty:
                st.success(f"👍 **{top_pos.iloc[0]['Feature']}** ({top_pos.iloc[0]['Feature Value']:.1f}) adds +${top_pos.iloc[0]['Impact (SHAP)']:.0f} to value.")
            
            if not top_neg.empty:
                st.error(f"👎 **{top_neg.iloc[0]['Feature']}** ({top_neg.iloc[0]['Feature Value']:.1f}) subtracts ${abs(top_neg.iloc[0]['Impact (SHAP)']):.0f} from value.")

# --- ROI Simulator ---
elif page == "ROI Simulator":
    st.title("Campaign ROI Simulator & Optimizer")
    
    st.markdown("Estimate impact and optimize budget allocation.")
    
    with st.expander("ℹ️ Model Assumptions & Industry Benchmarks"):
        st.markdown("""
        **About this Model:**
        *   **Correlation vs Causation:** CLV predictions are based on historical correlations. Future behavior is estimated, not guaranteed. The model identifies *who* is likely to be valuable, but does not inherently claim *why* (causality).
        *   **Simulation Nature:** The ROI calculator is a **stochastic simulation**. It projects outcomes based on your input assumptions (Conversion Rate, Uplift). It is *not* a retrospective measurement of past campaigns.
        *   **Experimental Validation:** We strongly recommend validating the `Uplift Factor` through randomized A/B tests before large-scale deployment.
        *   **Simplifying Assumptions:** The ROI calculation assumes a linear relationship between uplift and conversion within the specified ranges. External market factors (seasonality, competition) are not explicitly modeled in the simulation.
        
        **Typical Benchmarks (B2C E-commerce):**
        *   **Conversion Rate:** 2% - 5% for email campaigns; 0.5% - 2% for display ads.
        *   **Revenue Uplift:** Successful campaigns typically see 10-30% uplift (Factor 1.1 - 1.3).
        *   **Cost per Action:** Varies widely by channel (e.g., Email < $0.10, Facebook Ads $5-$20).
        """)

    mode = st.radio("Mode", ["Segment Simulation", "Decision Optimization (AI)"], horizontal=True)
    
    if mode == "Segment Simulation":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target_segment = st.selectbox("Target Segment", df['segment_label'].unique())
            budget = st.number_input("Campaign Budget ($)", value=5000, step=500)
            cost_per_action = st.number_input("Cost per Action ($)", value=5.00, step=0.50)
            conversion_rate = st.slider("Est. Conversion Rate", 0.0, 0.5, 0.05, step=0.01)
            uplift_factor = st.slider("Revenue Uplift Factor (e.g. 1.1 = +10%)", 1.0, 1.5, 1.2, step=0.05)
            
        with col2:
            results = simulate_campaign(df, target_segment, budget, cost_per_action, conversion_rate, uplift_factor)
            
            st.subheader("Simulation Results")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Targeted Customers", f"{results['targeted_count']} / {results['total_in_segment']}")
            m2.metric("Total Cost", f"${results['cost']:,.2f}")
            m3.metric("Est. ROI", f"{results['roi']:.1%}", delta_color="normal")
            
            st.write(f"**Base Revenue:** ${results['base_revenue']:,.2f}")
            st.write(f"**New Revenue:** ${results['new_revenue']:,.2f}")
            st.write(f"**Net Revenue Uplift:** ${results['revenue_uplift']:,.2f}")

    elif mode == "Decision Optimization (AI)":
        st.info("The AI will select the **optimal individual customers** to target to maximize revenue under your budget.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            budget = st.number_input("Total Budget ($)", value=10000, step=1000)
            cost_per_action = st.number_input("Cost per Action ($)", value=5.00, step=0.50)
            conversion_rate = st.slider("Est. Conversion Rate", 0.0, 0.5, 0.05, step=0.01)
            uplift_factor = st.slider("Revenue Uplift Factor", 1.0, 1.5, 1.2, step=0.05)
            
            if st.button("Run Optimization"):
                res = optimize_budget_allocation(df, budget, cost_per_action, conversion_rate, uplift_factor)
                st.session_state['opt_res'] = res
                
                # Frontier for 'What-If'
                frontier = calculate_efficient_frontier(df, budget * 2, cost_per_action, conversion_rate, uplift_factor)
                st.session_state['frontier'] = frontier
                
        with col2:
            if 'opt_res' in st.session_state:
                res = st.session_state['opt_res']
                
                st.subheader("Optimization Results")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Revenue Uplift", f"${res['revenue_uplift']:,.2f}", 
                          f"vs Random: ${res['revenue_uplift'] - res['random_uplift']:,.0f}")
                
                m2.metric("ROI", f"{res['roi']:.1%}", 
                          f"vs Random: {(res['roi'] - res['random_roi']):.1%}")
                          
                m3.metric("Customers Targeted", f"{res['targeted_count']}")
                
                # Charts
                tab1, tab2 = st.tabs(["Efficient Frontier", "Allocation Strategy"])
                
                with tab1:
                    frontier = st.session_state['frontier']
                    fig_front = px.line(frontier, x='Budget', y='Revenue Uplift', markers=True, 
                                        title="Efficient Frontier: Diminishing Returns")
                    # Add current point
                    fig_front.add_trace(go.Scatter(x=[budget], y=[res['revenue_uplift']], 
                                                   mode='markers', marker={'color': 'red', 'size': 12}, name='Current Allocation'))
                    st.plotly_chart(fig_front, use_container_width=True)
                    
                with tab2:
                    st.write("**Recommended Target Segments**")
                    fig_alloc = px.bar(res['allocation'], x='Count', y='Segment', orientation='h', title="Optimal Allocation by Segment")
                    st.plotly_chart(fig_alloc, use_container_width=True)
                    
                    st.write("*Note: Even within segments, the AI selects specific high-potential individuals.*")

        # --- Sensitivity Analysis ---
        if 'opt_res' in st.session_state:
            st.markdown("---")
            st.subheader("Campaign Sensitivity Analysis")
            st.markdown("Analyze how profitability (ROI) changes under different assumptions.")
            
            # Sensitivity Parameters
            sens_cols = st.columns(4)
            with sens_cols[0]:
                st.markdown("**Parameter Ranges**")
            with sens_cols[1]:
                s_conv_max = st.number_input("Max Conversion Rate", 0.1, 1.0, 0.3, step=0.05)
            with sens_cols[2]:
                s_uplift_max = st.number_input("Max Uplift Factor", 1.1, 3.0, 1.5, step=0.1)
            with sens_cols[3]:
                pass # Spacer
                
            if st.button("Run Sensitivity Analysis"):
                with st.spinner("Simulating Scenarios..."):
                    # Prepare ranges
                    ranges = {
                        'conversion_rate': np.linspace(0.01, s_conv_max, 20),
                        'uplift_factor': np.linspace(1.0, s_uplift_max, 20)
                        # Cost and Budget use defaults relative to current inputs
                    }
                    
                    base_params = {
                        'budget': budget,
                        'cost_per_action': cost_per_action,
                        'conversion_rate': conversion_rate,
                        'uplift_factor': uplift_factor
                    }
                    
                    sens_results = perform_sensitivity_analysis(df, base_params, ranges)
                    
                    # Store in session state to avoid re-running
                    st.session_state['sens_results'] = sens_results
            
            if 'sens_results' in st.session_state:
                res = st.session_state['sens_results']
                
                tab_s1, tab_s2, tab_s3 = st.tabs(["ROI vs Conversion", "ROI vs Uplift", "ROI vs Cost"])
                
                # Helper for charts
                def plot_sensitivity(data, x_col, title, current_val):
                    fig = px.line(data, x='Value', y='ROI', title=title)
                    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
                    
                    # Highlight current selection
                    current_roi = st.session_state['opt_res']['roi']
                    fig.add_trace(go.Scatter(x=[current_val], y=[current_roi], 
                                            mode='markers', marker={'color': 'green', 'size': 10}, 
                                            name='Current Scenario'))
                    return fig

                with tab_s1:
                    fig1 = plot_sensitivity(res['conversion_rate'], 'Value', "ROI Sensitivity to Conversion Rate", conversion_rate)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Interpretation
                    breakeven_conv = res['conversion_rate'][res['conversion_rate']['ROI'] >= 0]['Value'].min()
                    if pd.notna(breakeven_conv):
                        st.info(f"**Insight:** You need a conversion rate of at least **{breakeven_conv:.1%}** to be profitable.")
                    else:
                        st.warning("No profit within this range.")

                with tab_s2:
                    fig2 = plot_sensitivity(res['uplift_factor'], 'Value', "ROI Sensitivity to Uplift Factor", uplift_factor)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    breakeven_up = res['uplift_factor'][res['uplift_factor']['ROI'] >= 0]['Value'].min()
                    if pd.notna(breakeven_up):
                        st.info(f"**Insight:** Uplift factor needs to be above **{breakeven_up:.2f}x** for profitability.")

                with tab_s3:
                    fig3 = plot_sensitivity(res['cost_per_action'], 'Value', "ROI Sensitivity to Cost per Action", cost_per_action)
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # For cost, ROI goes down as cost goes up.
                    profitable_costs = res['cost_per_action'][res['cost_per_action']['ROI'] >= 0]
                    if not profitable_costs.empty:
                        max_cost = profitable_costs['Value'].max()
                        st.info(f"**Insight:** Cost per Action must be below **${max_cost:.2f}** to maintain positive ROI.")

        # --- Monte Carlo Simulation ---
        if 'opt_res' in st.session_state:
            st.markdown("---")
            st.subheader("Monte Carlo Risk Analysis")
            st.markdown("Assess the risk of campaign failure by simulating 1000+ potential outcomes.")
            
            mc_cols = st.columns(4)
            with mc_cols[0]:
                st.markdown("**Uncertainty (Std Dev)**")
            with mc_cols[1]:
                std_conv = st.number_input("Conversion uncertainty", 0.0, 0.2, 0.05, step=0.01, format="%.2f")
            with mc_cols[2]:
                std_uplift = st.number_input("Uplift uncertainty", 0.0, 0.5, 0.1, step=0.01, format="%.2f")
            with mc_cols[3]:
                n_sims = st.slider("Simulations", 100, 5000, 1000, 100)
                
            if st.button("Run Risk Simulation"):
                with st.spinner(f"Running {n_sims} simulations..."):
                    # Params
                    base_params = {
                        'budget': budget,
                        'cost_per_action': cost_per_action,
                        'conversion_rate': conversion_rate,
                        'uplift_factor': uplift_factor
                    }
                    std_devs = {
                        'conversion_rate': std_conv,
                        'uplift_factor': std_uplift
                    }
                    
                    mc_results = perform_monte_carlo_simulation(df, base_params, std_devs, n_sims)
                    st.session_state['mc_results'] = mc_results
            
            if 'mc_results' in st.session_state:
                mc = st.session_state['mc_results']
                
                if mc.empty:
                    st.warning("No customers selected in base scenario. Cannot simulate.")
                else:
                    # Metrics
                    mean_roi = mc['ROI'].mean()
                    prob_success = (mc['ROI'] > 0).mean()
                    var_5 = np.percentile(mc['ROI'], 5) # 5th percentile (Worst case usually)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Expected Mean ROI", f"{mean_roi:.1%}")
                    m2.metric("Prob. of Profit", f"{prob_success:.1%}")
                    m3.metric("Value at Risk (5th %)", f"{var_5:.1%}", delta="Worst Case Scenario", delta_color="off")
                    
                    # Histogram
                    st.subheader("Distribution of Possible ROI Outcomes")
                    
                    fig_hist = px.histogram(mc, x='ROI', nbins=50, title="ROI Probability Distribution",
                                           color_discrete_sequence=['#636EFA'])
                    
                    # Add mean and 0 line
                    fig_hist.add_vline(x=mean_roi, line_dash="dash", line_color="green", annotation_text="Mean")
                    fig_hist.add_vline(x=0, line_width=2, line_color="red", annotation_text="Break-even")
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Interpretation
                    if prob_success > 0.9:
                        st.success("Running this campaign is **Safe**. There is high certainty of positive return.")
                    elif prob_success > 0.6:
                        st.success("Running this campaign is **Likely Profitable**, but carries some risk.")
                    else:
                        st.error("Running this campaign is **Risky**. There is a significant chance of losing money.")
