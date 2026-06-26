import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Multi-Armed Bandit", page_icon="🎰", layout="wide")
st.title("🎰 Multi-Armed Bandit - Dynamic Pricing")
st.markdown("### Reinforcement Learning for Price Optimization")

data = load_data()
sales = data['sales'].copy()

# ============================================
# CONCEPT
# ============================================
st.info("""
**Multi-Armed Bandit (MAB) for Pricing:**
- 🎯 Each "arm" = a pricing strategy
- 💰 "Reward" = revenue/profit per arm
- 🔄 Algorithm learns which pricing maximizes revenue
- ⚖️ Balances **exploration** (try new prices) vs **exploitation** (use best known)
- 📈 Continuously adapts to market changes
""")

# ============================================
# BANDIT ALGORITHMS
# ============================================
class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n-1) * self.values[arm] + reward) / n


class UCB1Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0
    
    def select_arm(self):
        if 0 in self.counts:
            return np.argmax(self.counts == 0)
        ucb = self.values + np.sqrt(2 * np.log(self.total_count) / self.counts)
        return np.argmax(ucb)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_count += 1
        n = self.counts[arm]
        self.values[arm] = ((n-1) * self.values[arm] + reward) / n


class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
    
    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        # Normalize reward to [0,1]
        reward_norm = (reward - self.min_reward) / (self.max_reward - self.min_reward + 1e-9)
        reward_norm = np.clip(reward_norm, 0, 1)
        self.alpha[arm] += reward_norm
        self.beta[arm] += (1 - reward_norm)


# ============================================
# SIMULATE BANDITS FOR PRICING
# ============================================
st.subheader("🎮 Simulate Pricing Optimization")

# Define pricing arms (discount levels)
price_arms = {
    'No Discount (0%)': 0.0,
    'Low Discount (5%)': 0.05,
    'Medium Discount (10%)': 0.10,
    'High Discount (15%)': 0.15,
    'Very High (20%)': 0.20,
    'Extreme (25%)': 0.25
}

st.write("**Pricing Strategies (Arms):**")
st.dataframe(pd.DataFrame([{'Strategy': k, 'Discount': f"{v*100}%"} for k,v in price_arms.items()]),
             use_container_width=True)

# Configuration
c1, c2 = st.columns(2)
algorithm = c1.selectbox("Algorithm", ["Epsilon-Greedy", "UCB1", "Thompson Sampling"])
n_rounds = c2.slider("Simulation Rounds", 100, 5000, 1000)

# True reward probabilities (hidden from algorithm)
# In real world, these would be learned from market response
true_conversion = np.array([0.15, 0.18, 0.22, 0.25, 0.23, 0.20])  # Different conversion rates per price
true_aov = np.array([800, 780, 750, 700, 650, 600])  # Higher discount = lower AOV
true_revenue = true_conversion * true_aov  # Expected revenue per customer

if st.button("▶️ Run Simulation"):
    n_arms = len(price_arms)
    
    if algorithm == "Epsilon-Greedy":
        bandit = EpsilonGreedyBandit(n_arms, epsilon=0.1)
    elif algorithm == "UCB1":
        bandit = UCB1Bandit(n_arms)
    else:  # Thompson
        bandit = ThompsonSamplingBandit(n_arms)
        bandit.min_reward = true_revenue.min()
        bandit.max_reward = true_revenue.max()
    
    rewards_history = []
    arm_pulls = []
    optimal_arm = np.argmax(true_revenue)
    
    cumulative_reward = 0
    
    for t in range(n_rounds):
        arm = bandit.select_arm()
        # Reward with noise
        reward = np.random.normal(true_revenue[arm], 50)
        reward = max(0, reward)
        
        bandit.update(arm, reward)
        cumulative_reward += reward
        rewards_history.append(cumulative_reward)
        arm_pulls.append(arm)
    
    # Results
    st.markdown("---")
    st.subheader("📊 Results")
    
    arm_names = list(price_arms.keys())
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"₹{cumulative_reward/1e5:.1f} L")
    c2.metric("Best Arm Selected", arm_names[np.argmax(bandit.values if hasattr(bandit, 'values') else bandit.alpha/(bandit.alpha+bandit.beta))])
    c3.metric("Optimal Strategy", arm_names[optimal_arm])
    
    # Plots
    c1, c2 = st.columns(2)
    
    with c1:
        # Cumulative reward
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rewards_history, mode='lines', name='Algorithm Performance'))
        # Optimal baseline
        optimal_reward = [true_revenue[optimal_arm] * (i+1) for i in range(n_rounds)]
        fig.add_trace(go.Scatter(y=optimal_reward, mode='lines', name='Optimal',
                                  line=dict(dash='dash', color='red')))
        fig.update_layout(title='Cumulative Reward Over Time',
                          xaxis_title='Round', yaxis_title='Cumulative Revenue (₹)',
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        # Arm selection distribution
        pull_counts = pd.Series(arm_pulls).value_counts().sort_index()
        pull_df = pd.DataFrame({
            'Strategy': [arm_names[i] for i in pull_counts.index],
            'Times Selected': pull_counts.values,
            'True Reward': [true_revenue[i] for i in pull_counts.index]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pull_df['Strategy'], y=pull_df['Times Selected'],
                             name='Times Selected', marker_color='lightblue'))
        fig.add_trace(go.Scatter(x=pull_df['Strategy'], y=pull_df['True Reward']/10,
                                  name='True Revenue (scaled)', mode='lines+markers',
                                  yaxis='y2', line=dict(color='red', width=3)))
        fig.update_layout(title='Arm Selection vs True Reward',
                          xaxis_tickangle=-45,
                          yaxis=dict(title='Pull Count'),
                          yaxis2=dict(title='True Revenue', overlaying='y', side='right'),
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Regret analysis
    optimal_total = true_revenue[optimal_arm] * n_rounds
    regret = optimal_total - cumulative_reward
    regret_pct = (regret / optimal_total) * 100
    
    st.metric("Total Regret", f"₹{regret:,.0f}", f"{regret_pct:.2f}% of optimal")
    
    if regret_pct < 5:
        st.success(f"✅ Excellent! Algorithm converged close to optimal ({regret_pct:.2f}% regret)")
    elif regret_pct < 15:
        st.warning(f"⚠️ Good performance ({regret_pct:.2f}% regret)")
    else:
        st.info(f"📊 Algorithm still exploring ({regret_pct:.2f}% regret)")

# ============================================
# COMPARISON OF ALGORITHMS
# ============================================
st.markdown("---")
st.subheader("⚖️ Algorithm Comparison")

st.write("""
| Algorithm | Strategy | Best For |
|-----------|----------|----------|
| **Epsilon-Greedy** | Explore randomly with ε probability | Simple, stable environments |
| **UCB1** | Optimism in face of uncertainty | Fast convergence |
| **Thompson Sampling** | Bayesian posterior sampling | Best empirical performance |
""")

st.info("""
**Real-World Applications for FMCG CFO:**
- 💰 **Dynamic Pricing**: Find optimal price point for each SKU
- 🎯 **Promo Selection**: Which promotion gives best ROI
- 📺 **Ad Spend Allocation**: Distribute across channels
- 🏪 **Inventory Allocation**: Which products to stock where
- 📧 **Marketing Channel**: Email vs SMS vs WhatsApp
""")
