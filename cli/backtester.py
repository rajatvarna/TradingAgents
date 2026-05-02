import datetime
import pandas as pd
import yfinance as yf
from pathlib import Path
from rich.console import Console
from rich.table import Table

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.rating import parse_rating
from cli.stats_handler import StatsCallbackHandler

console = Console()

def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    selected_analysts: list,
    config: dict,
    output_dir: str = "backtests"
):
    """Run the TradingAgentsGraph iteratively over a date range."""
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate business days
    dates = pd.date_range(start=start, end=end, freq='B')
    
    results = []
    
    console.print(f"[bold green]Starting backtest for {ticker} from {start_date} to {end_date}[/bold green]")
    
    stats_handler = StatsCallbackHandler(provider=config.get("llm_provider", "google"))
    
    # Initialize graph once to avoid recompilation
    graph = TradingAgentsGraph(
        selected_analysts,
        config=config,
        debug=False,
        callbacks=[stats_handler],
    )
    
    for current_date in dates:
        date_str = current_date.strftime("%Y-%m-%d")
        console.print(f"Processing date: {date_str}...")
        
        final_state = None
        try:
            final_state, _ = graph.propagate(ticker, date_str)

            decision_text = final_state.get("final_trade_decision", "")
            rating = parse_rating(decision_text)
            confidence = TradingAgentsGraph._extract_confidence(decision_text)

            results.append({
                "date": date_str,
                "ticker": ticker,
                "rating": rating,
                "confidence": confidence,
            })
            console.print(f"  Decision: {rating} (Confidence: {confidence})")
            
        except Exception as e:
            console.print(f"[red]Error on {date_str}: {e}[/red]")
            results.append({
                "date": date_str,
                "ticker": ticker,
                "rating": "Error",
                "confidence": 0.0,
            })
            
    # Calculate returns
    df = pd.DataFrame(results)
    
    try:
        ticker_data = yf.Ticker(ticker)
        # Fetch an extra few days to get the forward return
        hist = ticker_data.history(start=start, end=end + datetime.timedelta(days=7))
        hist.index = hist.index.strftime('%Y-%m-%d')
        
        # Calculate 1-day forward return
        hist['Next_Close'] = hist['Close'].shift(-1)
        hist['Return_1d'] = (hist['Next_Close'] - hist['Close']) / hist['Close']
        
        # Merge
        df = df.merge(hist[['Return_1d']], left_on='date', right_index=True, how='left')
        
        # Calculate portfolio return (simplified: Buy=1x, Hold=0, Sell=-1x)
        # Wait, if we just want a simple vector backtest:
        def calc_pos(rating):
            if rating in ["Buy", "Overweight"]: return 1.0
            if rating in ["Sell", "Underweight"]: return -1.0
            return 0.0
            
        df['Position'] = df['rating'].apply(calc_pos)
        df['Strategy_Return'] = df['Position'] * df['Return_1d']
        df['Cumulative_Return'] = (1 + df['Strategy_Return'].fillna(0)).cumprod() - 1
        
    except Exception as e:
        console.print(f"[red]Error fetching returns for backtest: {e}[/red]")
        
    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / f"{ticker}_backtest_{start_date}_{end_date}.csv"
    df.to_csv(csv_file, index=False)
    console.print(f"[bold green]Backtest completed. Results saved to {csv_file}[/bold green]")
    
    # Display summary
    table = Table(title=f"Backtest Summary: {ticker}")
    table.add_column("Total Days")
    table.add_column("Buys")
    table.add_column("Sells")
    table.add_column("Cumulative Return")
    table.add_column("Total Cost")
    
    buys = len(df[df['Position'] > 0])
    sells = len(df[df['Position'] < 0])
    cum_ret = f"{df['Cumulative_Return'].iloc[-1]:.2%}" if 'Cumulative_Return' in df.columns else "N/A"
    
    table.add_row(str(len(df)), str(buys), str(sells), cum_ret, f"${stats_handler.total_cost:.2f}")
    console.print(table)
    
    return csv_file
