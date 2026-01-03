import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import warnings
import itertools

warnings.filterwarnings('ignore')

nltk.download('vader_lexicon')

if 'portfolio_tickers' not in st.session_state:
    st.session_state.portfolio_tickers = []

class PortfolioAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis")
        except:
            self.sentiment_pipeline = None

    def get_stock_data(self, tickers, period="3mo"):
        data = {}
        for ticker in tickers:
            try:
                stock_data = yf.download(ticker, period=period, progress=False)
                if not stock_data.empty:
                    data[ticker] = stock_data
            except:
                pass
        return data

    def calculate_technical_indicators(self, data):
        indicators = {}

        for ticker, df in data.items():
            if df.empty:
                continue

            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50)

            indicators[ticker] = df

        return indicators

    def analyze_fundamentals(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals = {
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'price_to_book': info.get('priceToBook', 'N/A'),
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'return_on_equity': info.get('returnOnEquity', 'N/A'),
                'profit_margins': info.get('profitMargins', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'sector': info.get('sector', 'N/A')
            }

            return fundamentals
        except:
            return {}

    def get_sentiment(self, ticker):
        """Simple price-based sentiment proxy."""
        try:
            stock_data = yf.download(ticker, period="1mo", progress=False, multi_level_index=False)

            stock_data = stock_data.dropna(subset=['Close'])

            if stock_data.empty:
                return 0.0

            close_last = stock_data['Close'].iloc[-1].item()
            close_first = stock_data['Close'].iloc[0].item()

            if close_first == 0:
                return 0.0

            recent_return = (close_last / close_first) - 1
            sentiment = np.tanh(recent_return * 10)

            return float(sentiment)
        except Exception as e:
            return 0.0

    def get_vix_data(self):
        try:
            vix_data = yf.download('^VIX', period='1mo', progress=False, multi_level_index=False)
            vix_data = vix_data.dropna(subset=['Close'])

            if not vix_data.empty:
                return vix_data['Close'].iloc[-1].item()
            return 20.0
        except Exception as e:
            return 20.0

    def calculate_market_temperature(self, tickers):
        try:
            sentiments = []
            for ticker in tickers:
                sentiment = self.get_sentiment(ticker)
                sentiments.append(sentiment)


            if not sentiments:
                avg_sentiment = 0.0
            else:
                avg_sentiment = np.mean(sentiments)

            vix = self.get_vix_data()

            vix_sentiment = -((vix - 15) / 30)

            market_temp = (avg_sentiment * 0.6) + (vix_sentiment * 0.4)


            if market_temp > 0.2:
                condition = "Bullish"
                color = "green"
                emoji = "📈"
            elif market_temp > 0.05:
                condition = "Slightly Bullish"
                color = "lightgreen"
                emoji = "↗️"
            elif market_temp < -0.2:
                condition = "Bearish"
                color = "red"
                emoji = "📉"
            elif market_temp < -0.05:
                condition = "Slightly Bearish"
                color = "pink"
                emoji = "↘️"
            else:
                condition = "Neutral"
                color = "yellow"
                emoji = "➡️"

            return {
                'temperature': market_temp,
                'condition': condition,
                'color': color,
                'emoji': emoji,
                'vix': vix,
                'avg_sentiment': avg_sentiment
            }
        except:
            return {'temperature': 0, 'condition': 'Neutral', 'color': 'yellow', 'emoji': '➡️', 'vix': 20, 'avg_sentiment': 0}

    def calculate_portfolio_metrics(self, prices_df):
        try:
            returns_df = prices_df.pct_change().dropna()

            if returns_df.empty:
                return {"volatility": 0, "sharpe_ratio": 0, "max_drawdown": 0, "risk_rating": 5}

            volatility = returns_df.std().mean() * np.sqrt(252)

            sharpe_ratio = (
                returns_df.mean().mean() / returns_df.std().mean() * np.sqrt(252)
                if returns_df.std().mean() > 0 else 0
            )

            cumulative_returns = (1 + returns_df).cumprod()
            max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min().mean()

            risk_score = min(volatility / 0.4, 1.0)
            risk_rating = int(risk_score * 9) + 1

            return {
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "risk_rating": risk_rating
            }
        except:
            return {"volatility": 0, "sharpe_ratio": 0, "max_drawdown": 0, "risk_rating": 5}


def main():
    st.set_page_config(
        page_title="Portfolio Doctor",
        page_icon="🩺📈",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🩺📈 Portfolio Doctor</h1>', unsafe_allow_html=True)

    analyzer = PortfolioAnalyzer()

    # -------- SIDEBAR: PORTFOLIO SETUP ----------
    with st.sidebar:
        st.header("Portfolio Setup")

        ticker_input = st.text_input("Enter Stock Ticker:").upper()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Stock") and ticker_input:
                if len(st.session_state.portfolio_tickers) < 10:
                    if ticker_input not in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers.append(ticker_input)
                else:
                    st.error("Max 10 stocks")

        with col2:
            if st.button("Clear All"):
                st.session_state.portfolio_tickers = []

        if st.session_state.portfolio_tickers:
            st.subheader("Current Portfolio")
            for i, ticker in enumerate(st.session_state.portfolio_tickers):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"{i+1}. {ticker}")
                with c2:
                    if st.button("Remove", key=f"del_{ticker}"):
                        st.session_state.portfolio_tickers.remove(ticker)
                        st.rerun()

    if not st.session_state.portfolio_tickers:
        st.info("Add stocks to begin analysis")
        return

    # -------- CORE DATA FETCH & METRICS ----------
    with st.spinner("Analyzing portfolio..."):
        data = analyzer.get_stock_data(st.session_state.portfolio_tickers)

        if len(data) == 0:
            st.error("No data retrieved")
            return

        prices_df = pd.DataFrame()
        for ticker in st.session_state.portfolio_tickers:
            if ticker in data and not data[ticker].empty:
                prices_df[ticker] = data[ticker]['Close']

        if prices_df.empty:
            st.error("No price data")
            return

        technical_data = analyzer.calculate_technical_indicators(data)
        market_temp = analyzer.calculate_market_temperature(st.session_state.portfolio_tickers)
        portfolio_metrics = analyzer.calculate_portfolio_metrics(prices_df)
        returns_df = prices_df.pct_change().dropna()

    # -------- TOP METRICS STRIP ----------
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Market Temperature",
            f"{market_temp['emoji']} {market_temp['condition']}",
            f"{market_temp['temperature']:.3f}"
        )
        st.write(f"VIX: {market_temp['vix']:.1f}")

    with col2:
        risk_rating = portfolio_metrics["risk_rating"]
        st.metric("Portfolio Risk", f"{risk_rating}/10")
        st.progress(risk_rating / 10)
        st.write(f"Volatility: {portfolio_metrics['volatility']:.2%}")

    with col3:
        st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown']:.2%}")

    with col4:
        st.metric("Portfolio Size", f"{len(st.session_state.portfolio_tickers)} stocks")
        st.metric("Data Points", f"{len(prices_df)} days")

    # -------- TABS --------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Portfolio Overview",
        "Stock Analysis",
        "Fundamentals",
        "Correlation & Diversification",
        "Rebalancing Advisor",
        "Stress Test"
    ])

    # ========== TAB 1: Portfolio Overview ==========
    with tab1:
        st.subheader("Portfolio Performance")

        normalized_prices = prices_df / prices_df.iloc[0]

        fig = go.Figure()
        for ticker in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker],
                name=ticker,
                mode='lines'
            ))

        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.write("Recent Performance:")
            recent_returns = (prices_df.iloc[-1] / prices_df.iloc[0] - 1) * 100
            for ticker, ret in recent_returns.items():
                st.write(f"{ticker}: {ret:+.2f}%")

        with c2:
            st.write("Current Allocations:")
            current_values = prices_df.iloc[-1]
            total_value = current_values.sum()
            for ticker, value in current_values.items():
                allocation = (value / total_value) * 100
                st.write(f"{ticker}: {allocation:.1f}%")

    # ========== TAB 2: Multi-Stock Technical Analysis ==========
    with tab2:
        st.subheader("Individual Stock Analysis (Multi-Select)")

        selected_tickers = st.multiselect(
            "Select stocks to analyze:",
            st.session_state.portfolio_tickers,
            default=st.session_state.portfolio_tickers[:1]
        )

        if not selected_tickers:
            st.info("Select at least one stock to see analysis.")
        else:
            for sel in selected_tickers:
                if sel not in technical_data:
                    continue

                stock_data = technical_data[sel]
                st.markdown(f"### {sel}")

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=(f'{sel} Price', 'RSI'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )

                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=stock_data['Close'],
                    name='Close', line=dict(color='blue')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=stock_data['SMA_20'],
                    name='SMA 20', line=dict(color='orange')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=stock_data['SMA_50'],
                    name='SMA 50', line=dict(color='red')
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=stock_data['RSI'],
                    name='RSI', line=dict(color='purple')
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                current_rsi = float(stock_data['RSI'].iloc[-1])
                current_price = float(stock_data['Close'].iloc[-1])
                sma_20 = float(stock_data['SMA_20'].iloc[-1])

                c1, c2, c3 = st.columns(3)
                with c1:
                    if current_rsi > 70:
                        st.error("RSI Overbought")
                    elif current_rsi < 30:
                        st.success("RSI Oversold")
                    else:
                        st.info(f"RSI: {current_rsi:.1f}")

                with c2:
                    if current_price > sma_20:
                        st.success("Above SMA 20")
                    else:
                        st.error("Below SMA 20")

                with c3:
                    sentiment = analyzer.get_sentiment(sel)
                    if sentiment > 0.1:
                        st.success(f"Sentiment: {sentiment:.2f}")
                    elif sentiment < -0.1:
                        st.error(f"Sentiment: {sentiment:.2f}")
                    else:
                        st.info(f"Sentiment: {sentiment:.2f}")

                st.markdown("---")

    # ========== TAB 3: Fundamental Comparison (Multi-Stock) ==========
    with tab3:
        st.subheader("Fundamental Comparison (Multi-Stock)")

        selected_funds = st.multiselect(
            "Select stocks for fundamental comparison:",
            st.session_state.portfolio_tickers,
            default=st.session_state.portfolio_tickers[: min(3, len(st.session_state.portfolio_tickers))]
        )

        if not selected_funds:
            st.info("Select at least one stock to see fundamentals.")
        else:
            rows = [
                ('P/E Ratio', 'pe_ratio'),
                ('Forward P/E', 'forward_pe'),
                ('Price/Book', 'price_to_book'),
                ('Debt/Equity', 'debt_to_equity'),
                ('ROE', 'return_on_equity'),
                ('Profit Margin', 'profit_margins'),
                ('Dividend Yield', 'dividend_yield'),
                ('Beta', 'beta'),
                ('Market Cap', 'market_cap'),
                ('Sector', 'sector')
            ]

            table = []
            for label, key in rows:
                row = {'Metric': label}
                for ticker in selected_funds:
                    f = analyzer.analyze_fundamentals(ticker)
                    val = f.get(key, 'N/A')
                    if isinstance(val, float):
                        if key in ['profit_margins', 'dividend_yield', 'return_on_equity']:
                            row[ticker] = f"{val*100:.2f}%"
                        else:
                            row[ticker] = f"{val:.2f}"
                    else:
                        row[ticker] = val
                table.append(row)

            fund_df = pd.DataFrame(table)
            st.dataframe(fund_df)

    # ========== TAB 4: Correlation & Diversification ==========
    with tab4:
        st.subheader("Correlation & Diversification")

        if returns_df.empty or len(returns_df.columns) < 2:
            st.info("Need at least 2 stocks for correlation.")
        else:
            correlation_matrix = returns_df.corr()

            fig = px.imshow(
                correlation_matrix,
                title="Stock Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)

            pairs = list(itertools.combinations(correlation_matrix.columns, 2))
            abs_corrs = [abs(correlation_matrix.loc[a, b]) for a, b in pairs] if pairs else []
            avg_abs_corr = float(np.mean(abs_corrs)) if abs_corrs else 0.0

            diversification_score = int((1.0 - avg_abs_corr) * 100)
            diversification_score = max(0, min(100, diversification_score))

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Diversification Score", f"{diversification_score}/100")
                st.progress(diversification_score / 100.0)
            with c2:
                if diversification_score >= 70:
                    st.write("Your portfolio is well diversified.")
                elif diversification_score >= 40:
                    st.write("Your portfolio is moderately diversified.")
                else:
                    st.write("Your portfolio is poorly diversified; many holdings move together.")

    # ========== TAB 5: Rebalancing Advisor ==========
    with tab5:
        st.subheader("Rebalancing Advisor")

        latest_prices = prices_df.iloc[-1]
        total_value = latest_prices.sum()

        if total_value <= 0:
            st.info("Unable to compute weights.")
        else:
            auto_weights = latest_prices / total_value

            st.write("### Step 1: Enter or confirm your current weights (%)")
            use_manual = st.checkbox("Manually edit current weights", value=False)

            current_weights_pct = {}
            if use_manual:
                total_entered = 0.0
                for t in auto_weights.index:
                    default = float((auto_weights[t] * 100).round(2))
                    w = st.number_input(
                        f"{t} weight (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=default,
                        step=0.5
                    )
                    current_weights_pct[t] = w
                    total_entered += w

                st.write(f"Total entered: {total_entered:.2f}%")
                if abs(total_entered - 100.0) > 1e-6:
                    st.warning("Weights do not sum to 100%. They will be normalized internally.")
                weights_series = pd.Series(current_weights_pct) / max(total_entered, 1e-9)
            else:
                weights_series = auto_weights
                current_weights_pct = (weights_series * 100).round(2).to_dict()
                st.write("Using price-implied weights:")
                st.dataframe(pd.Series(current_weights_pct, name="Weight %"))

            st.write("### Step 2: Choose target allocation model")
            model_choice = st.radio(
                "Target Model",
                [
                    "Equal-Weight",
                    "Risk-Parity",
                    "Defensive (Low Volatility)",
                    "Aggressive (High Return)",
                    "Sector Diversification"
                ],
                index=0
            )

            if returns_df.empty:
                st.info("Not enough return data for advanced models. Using Equal-Weight.")
                ideal_weights = pd.Series(1.0 / len(weights_series), index=weights_series.index)
            else:
                vol = returns_df.std()
                mean_ret = returns_df.mean()

                if model_choice == "Equal-Weight":
                    ideal_weights = pd.Series(1.0 / len(weights_series), index=weights_series.index)

                elif model_choice == "Risk-Parity":
                    inv_vol = 1.0 / vol.replace(0, np.nan)
                    inv_vol = inv_vol.fillna(0)
                    if inv_vol.sum() == 0:
                        ideal_weights = pd.Series(1.0 / len(weights_series), index=weights_series.index)
                    else:
                        ideal_weights = inv_vol / inv_vol.sum()

                elif model_choice == "Defensive (Low Volatility)":
                    inv_vol2 = 1.0 / (vol.replace(0, np.nan) ** 2)
                    inv_vol2 = inv_vol2.fillna(0)
                    if inv_vol2.sum() == 0:
                        ideal_weights = pd.Series(1.0 / len(weights_series), index=weights_series.index)
                    else:
                        ideal_weights = inv_vol2 / inv_vol2.sum()

                elif model_choice == "Aggressive (High Return)":
                    pos_ret = mean_ret.clip(lower=0)
                    if pos_ret.sum() == 0:
                        ideal_weights = pd.Series(1.0 / len(weights_series), index=weights_series.index)
                    else:
                        ideal_weights = pos_ret / pos_ret.sum()

                else:  # Sector Diversification
                    sectors = {}
                    for t in weights_series.index:
                        f = analyzer.analyze_fundamentals(t)
                        sectors[t] = f.get('sector', 'Unknown')

                    sector_series = pd.Series(sectors)
                    sector_counts = sector_series.value_counts()
                    sector_target = pd.Series(1.0 / len(sector_counts), index=sector_counts.index)

                    ideal_weights = pd.Series(0.0, index=weights_series.index)
                    for sec, tickers_in_sec in sector_series.groupby(sector_series):
                        tickers_list = list(tickers_in_sec.index)
                        if len(tickers_list) == 0:
                            continue
                        w_each = sector_target.get(sec, 0.0) / len(tickers_list)
                        for t in tickers_list:
                            ideal_weights[t] = w_each

                    if ideal_weights.sum() > 0:
                        ideal_weights = ideal_weights / ideal_weights.sum()
                    else:
                        ideal_weights = pd.Series(1.0 / len(weights_series), index=weights_series.index)

            ideal_weights = ideal_weights.loc[weights_series.index]

            rebalance_df = pd.DataFrame({
                "Current weight (%)": (weights_series * 100).round(2),
                "Ideal weight (%)": (ideal_weights * 100).round(2)
            })
            rebalance_df["Difference (%)"] = (rebalance_df["Current weight (%)"]
                                              - rebalance_df["Ideal weight (%)"]).round(2)

            actions = []
            for diff in rebalance_df["Difference (%)"]:
                if diff > 2.0:
                    if diff > 5.0:
                        actions.append("Reduce exposure (strong overweight)")
                    else:
                        actions.append("Slight reduction")
                elif diff < -2.0:
                    if diff < -5.0:
                        actions.append("Increase exposure (strong underweight)")
                    else:
                        actions.append("Add gradually")
                else:
                    actions.append("Close to target")

            rebalance_df["Action"] = actions

            st.write("### Rebalancing Table")
            st.dataframe(rebalance_df)

            fig_diff = go.Figure()
            fig_diff.add_trace(go.Bar(
                x=rebalance_df.index,
                y=rebalance_df["Difference (%)"],
                marker_color=["red" if v > 0 else "green" for v in rebalance_df["Difference (%)"]],
                name="Over(+)/Under(-) vs Target"
            ))
            fig_diff.update_layout(
                title="Overweight / Underweight vs Target",
                yaxis_title="Difference (%)"
            )
            st.plotly_chart(fig_diff, use_container_width=True)

            # Rebalancing score (how close to target)
            l1_distance = (rebalance_df["Difference (%)"].abs().sum()) / 2.0
            max_possible = 100.0
            rebal_score = max(0, 100 - (l1_distance / max_possible) * 100)
            st.metric("Rebalancing Score (how close to target)", f"{rebal_score:.1f}/100")

            overweight = rebalance_df[rebalance_df["Difference (%)"] > 0]\
                .sort_values("Difference (%)", ascending=False)
            underweight = rebalance_df[rebalance_df["Difference (%)"] < 0]\
                .sort_values("Difference (%)")

            st.write("### AI-style Suggestions")
            bullet_points = []
            for idx, row in overweight.iterrows():
                bullet_points.append(f"{idx}: Overweight by {row['Difference (%)']:.1f}% → Reduce exposure")
            for idx, row in underweight.iterrows():
                bullet_points.append(f"{idx}: Underweight by {abs(row['Difference (%)']):.1f}% → Increase exposure")

            for line in bullet_points:
                st.write("- " + line)

            if not overweight.empty:
                top_ow = ", ".join(list(overweight.index[:2]))
            else:
                top_ow = "no major overweights"

            if not underweight.empty:
                top_uw = ", ".join(list(underweight.index[:2]))
            else:
                top_uw = "no major underweights"

            st.write("**AI Recommendation:**")
            st.write(
                f"Your portfolio shows higher concentration in {top_ow}. "
                f"Consider trimming these and reallocating some capital towards {top_uw} "
                "to achieve a more balanced allocation."
            )
            st.caption("Note: This is an educational tool, not financial advice.")

    # ========== TAB 6: Stress Test ==========
    with tab6:
        st.subheader("Portfolio Drawdown Stress Test")

        latest_prices = prices_df.iloc[-1]
        total_value = latest_prices.sum()

        if total_value <= 0:
            st.info("Unable to compute portfolio weights for stress test.")
        else:
            current_weights = latest_prices / total_value

            betas = {}
            for ticker in current_weights.index:
                fundamentals = analyzer.analyze_fundamentals(ticker)
                beta = fundamentals.get('beta') if fundamentals else None
                 if __name__ == "__main__":
    main()

