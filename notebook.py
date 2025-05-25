#NOMURA QUANT CHALLENGE 2025

#The format for the weights dataframe for the backtester is attached with the question.
#Complete the below codes wherever applicable

import pandas as pd
import numpy as np
import pickle


def backtester_without_TC(weights_df):
    #Update data file path here
    data = pd.read_csv('cross_val_data.csv')

    weights_df = weights_df.fillna(0)
    start_date = 3500
    end_date = 3999
    # print(weights_df.loc[1:1,1:1])
    weights_df = weights_df.drop(weights_df.columns[0], axis=1)
    weights_df.index = range(start_date, end_date+1)

    initial_notional = 1

    df_returns = pd.DataFrame()

    for i in range(0,20):
        data_symbol = data[data['Symbol']==i]
        data_symbol = data_symbol['Close']
        data_symbol = data_symbol.reset_index(drop=True)   
        data_symbol = data_symbol/data_symbol.shift(1) - 1
        df_returns =  pd.concat([df_returns,data_symbol], axis=1, ignore_index=True)
    

    df_returns.index = range(start_date, end_date+1)
    df_returns = df_returns.fillna(0)
    weights_df = weights_df.loc[start_date:end_date]    
    df_returns = df_returns.loc[start_date:end_date]

    array_mul=np.multiply(np.array(df_returns),np.array(weights_df))
    df_returns = pd.DataFrame(array_mul, columns=weights_df.columns, index=weights_df.index)
    
    notional = initial_notional

    returns = []
    
    for date in range(start_date,end_date+1):
        returns.append(df_returns.loc[date].values.sum())
        notional = notional * (1+returns[date-start_date])

    net_return = ((notional - initial_notional)/initial_notional)*100
    sharpe_ratio = (pd.DataFrame(returns).mean().values[0])/pd.DataFrame(returns).std().values[0]

    return [net_return, sharpe_ratio]

def task1_Strategy1():
    train = pd.read_csv('train_data.csv')
    val = pd.read_csv('cross_val_data.csv')
    weights = []
    for sym in range(20):
        df = pd.concat([train[train.Symbol==sym], val[val.Symbol==sym]], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df['weekly'] = df['Close'].pct_change(periods=5) * 100
        df['hist'] = df['weekly'].rolling(50).mean()
        for d in range(3500, 4000):
            val_ = df.loc[d, 'hist'] if d in df.index else np.nan
            weights.append((d, sym, val_))
    wdf = pd.DataFrame(weights, columns=['Date', 'Symbol', 'hist']).pivot(index='Date', columns='Symbol', values='hist').fillna(0)
    out = []
    for idx, row in wdf.iterrows():
        order = row.sort_values(ascending=False)
        w = np.zeros(20)
        w[order.index[:6]] = -1/6
        w[order.index[-6:]] = 1/6
        out.append([idx] + list(w))
    return pd.DataFrame(out, columns=[''] + [str(i) for i in range(20)])

def task1_Strategy2():
    train = pd.read_csv('train_data.csv')
    val = pd.read_csv('cross_val_data.csv')
    weights = []
    for sym in range(20):
        df = pd.concat([train[train.Symbol==sym], val[val.Symbol==sym]], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df['lma'] = df['Close'].rolling(30).mean()
        df['sma'] = df['Close'].rolling(5).mean()
        df['rel'] = 100 * (df['sma'] - df['lma']) / df['sma']
        for d in range(3500, 4000):
            val_ = df.loc[d, 'rel'] if d in df.index else np.nan
            weights.append((d, sym, val_))
    wdf = pd.DataFrame(weights, columns=['Date', 'Symbol', 'rel']).pivot(index='Date', columns='Symbol', values='rel').fillna(0)
    out = []
    for idx, row in wdf.iterrows():
        order = row.sort_values(ascending=False)
        w = np.zeros(20)
        w[order.index[:5]] = -1/5
        w[order.index[-5:]] = 1/5
        out.append([idx] + list(w))
    return pd.DataFrame(out, columns=[''] + [str(i) for i in range(20)])

def task1_Strategy3():
    train = pd.read_csv('train_data.csv')
    val = pd.read_csv('cross_val_data.csv')
    weights = []
    for sym in range(20):
        df = pd.concat([train[train.Symbol==sym], val[val.Symbol==sym]], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df['roc'] = 100 * (df['Close'] - df['Close'].shift(7)) / df['Close'].shift(7)
        for d in range(3500, 4000):
            val_ = df.loc[d, 'roc'] if d in df.index else np.nan
            weights.append((d, sym, val_))
    wdf = pd.DataFrame(weights, columns=['Date', 'Symbol', 'roc']).pivot(index='Date', columns='Symbol', values='roc').fillna(0)
    out = []
    for idx, row in wdf.iterrows():
        order = row.sort_values(ascending=False)
        w = np.zeros(20)
        w[order.index[:4]] = -0.25
        w[order.index[-4:]] = 0.25
        out.append([idx] + list(w))
    return pd.DataFrame(out, columns=[''] + [str(i) for i in range(20)])

def task1_Strategy4():
    train = pd.read_csv('train_data.csv')
    val = pd.read_csv('cross_val_data.csv')
    weights = []
    for sym in range(20):
        df = pd.concat([train[train.Symbol==sym], val[val.Symbol==sym]], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df['sma21'] = df['Close'].rolling(21).mean()
        df['std21'] = df['Close'].rolling(21).std()
        df['res'] = df['sma21'] + 3 * df['std21']
        df['sup'] = df['sma21'] - 3 * df['std21']
        df['pr'] = (df['Close'] - df['res']) / df['res'] * 100
        df['ps'] = (df['Close'] - df['sup']) / df['sup'] * 100
        for d in range(3500, 4000):
            pr = df.loc[d, 'pr'] if d in df.index else np.nan
            ps = df.loc[d, 'ps'] if d in df.index else np.nan
            weights.append((d, sym, pr, ps))
    wdf = pd.DataFrame(weights, columns=['Date', 'Symbol', 'pr', 'ps'])
    pr_df = wdf.pivot(index='Date', columns='Symbol', values='pr').fillna(0)
    ps_df = wdf.pivot(index='Date', columns='Symbol', values='ps').fillna(0)
    out = []
    for idx in pr_df.index:
        ps_row = ps_df.loc[idx]
        pr_row = pr_df.loc[idx]
        top_support = ps_row.nsmallest(4).index
        rest = list(set(range(20)) - set(top_support))
        top_resist = pr_row[rest].nlargest(4).index
        w = np.zeros(20)
        w[list(top_support)] = 0.25
        w[list(top_resist)] = -0.25
        out.append([idx] + list(w))
    return pd.DataFrame(out, columns=[''] + [str(i) for i in range(20)])

def task1_Strategy5():
    train = pd.read_csv('train_data.csv')
    val = pd.read_csv('cross_val_data.csv')
    weights = []
    for sym in range(20):
        df = pd.concat([train[train.Symbol==sym], val[val.Symbol==sym]], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df['low14'] = df['Low'].rolling(14).min()
        df['high14'] = df['High'].rolling(14).max()
        df['k'] = 100 * (df['Close'] - df['low14']) / (df['high14'] - df['low14'])
        for d in range(3500, 4000):
            val_ = df.loc[d, 'k'] if d in df.index else np.nan
            weights.append((d, sym, val_))
    wdf = pd.DataFrame(weights, columns=['Date', 'Symbol', 'k']).pivot(index='Date', columns='Symbol', values='k').fillna(0)
    out = []
    for idx, row in wdf.iterrows():
        order = row.sort_values()
        w = np.zeros(20)
        w[order.index[:3]] = 1/3
        w[order.index[-3:]] = -1/3
        out.append([idx] + list(w))
    return pd.DataFrame(out, columns=[''] + [str(i) for i in range(20)])


def task1():
    print("Starting Task 1...")
    Strategy1 = task1_Strategy1()
    Strategy2 = task1_Strategy2()
    Strategy3 = task1_Strategy3()
    Strategy4 = task1_Strategy4()
    Strategy5 = task1_Strategy5()

    print("Running backtest for Strategy 1...")
    performanceStrategy1 = backtester_without_TC(Strategy1)
    print("Running backtest for Strategy 2...")
    performanceStrategy2 = backtester_without_TC(Strategy2)
    print("Running backtest for Strategy 3...")
    performanceStrategy3 = backtester_without_TC(Strategy3)
    print("Running backtest for Strategy 4...")
    performanceStrategy4 = backtester_without_TC(Strategy4)
    print("Running backtest for Strategy 5...")
    performanceStrategy5 = backtester_without_TC(Strategy5)

    output_df = pd.DataFrame({'Strategy1':performanceStrategy1, 'Strategy2': performanceStrategy2, 'Strategy3': performanceStrategy3, 'Strategy4': performanceStrategy4, 'Strategy5': performanceStrategy5})
    print(f"\n===== Task 1 Results =====\n{output_df}")
    output_df.to_csv('task1.csv')
    print("Task 1 completed. Results saved to task1.csv")
    return



def task2():
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import seaborn as sns
    
    print("Starting Task 2...")
    strategies = [task1_Strategy1(), task1_Strategy2(), task1_Strategy3(), task1_Strategy4(), task1_Strategy5()]
    start_date, end_date = 3500, 3999
    data = pd.read_csv('cross_val_data.csv')
    all_returns = []
    strategy_names = ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4", "Strategy 5"]
    
    # Calculate returns for each strategy
    for strat in strategies:
        strat_ = strat.drop(strat.columns[0], axis=1)
        strat_.index = range(start_date, end_date+1)
        df_returns = pd.DataFrame()
        for i in range(20):
            data_symbol = data[data['Symbol'] == i]['Close'].reset_index(drop=True)
            data_symbol = data_symbol / data_symbol.shift(1) - 1
            df_returns = pd.concat([df_returns, data_symbol], axis=1, ignore_index=True)
        df_returns.index = range(start_date, end_date+1)
        df_returns = df_returns.fillna(0)
        strat_ = strat_.loc[start_date:end_date]
        df_returns = df_returns.loc[start_date:end_date]
        daily_returns = (strat_.values * df_returns.values).sum(axis=1)
        all_returns.append(daily_returns)
    all_returns = np.array(all_returns)  # shape: (5, 500)
    
    # Calculate cumulative returns for each strategy for plotting
    cum_returns = np.zeros((5, end_date - start_date + 1))
    for s in range(5):
        cum_returns[s, 0] = 1.0  # Start with $1
        for d in range(1, end_date - start_date + 1):
            cum_returns[s, d] = cum_returns[s, d-1] * (1 + all_returns[s, d])
    
    # Plotting the cumulative returns for each strategy
    plt.figure(figsize=(12, 6))
    for s in range(5):
        plt.plot(range(start_date, end_date + 1), cum_returns[s], label=strategy_names[s])
    plt.title('Cumulative Returns of Individual Strategies')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('strategy_cumulative_returns.png')
    
    # Calculate rolling Sharpe ratios (20-day window)
    rolling_window = 20
    rolling_sharpe = []
    rolling_dates = []

    for s in range(5):
        strategy_sharpes = []
        for d in range(rolling_window, end_date - start_date + 1):
            window_returns = all_returns[s, d-rolling_window:d]
            strategy_sharpes.append(window_returns.mean() / (window_returns.std() + 1e-8))
            if s == 0:  # Only collect dates once
                rolling_dates.append(start_date + d)
        rolling_sharpe.append(strategy_sharpes)

    # Convert to numpy arrays after collection
    rolling_sharpe = np.array(rolling_sharpe)

    # Plot rolling Sharpe ratios
    plt.figure(figsize=(12, 6))
    for s in range(5):
        plt.plot(rolling_dates, rolling_sharpe[s], label=strategy_names[s])
    plt.title(f'{rolling_window}-Day Rolling Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rolling_sharpe_ratios.png')

    output_weights = []
    chosen_strat_idx = []
    for day in range(end_date - start_date + 1):
        if day < 10:
            idx = 0  # Use first strategy for first 10 days
        else:
            recent_perf = all_returns[:, day-10:day].mean(axis=1)
            idx = np.argmax(recent_perf)
        chosen_strat_idx.append(idx)
        weights_row = [start_date + day] + list(strategies[idx].iloc[day, 1:].values)
        output_weights.append(weights_row)
    
    # Visualize strategy selection frequency
    strategy_counts = np.bincount(chosen_strat_idx, minlength=5)
    plt.figure(figsize=(10, 6))
    plt.bar(strategy_names, strategy_counts)
    plt.title('Strategy Selection Frequency')
    plt.ylabel('Number of Days Selected')
    plt.tight_layout()
    plt.savefig('strategy_selection_frequency.png')
    
    # Visualize strategy selection over time
    plt.figure(figsize=(12, 6))
    plt.plot(range(start_date, end_date + 1), chosen_strat_idx)
    plt.yticks(range(5), strategy_names)
    plt.title('Strategy Selection Over Time')
    plt.xlabel('Date')
    plt.ylabel('Selected Strategy')
    plt.grid(True, alpha=0.3)
    plt.savefig('strategy_selection_timeline.png')
    
    # Track combined strategy performance
    combined_returns = np.zeros(end_date - start_date + 1)
    for day in range(end_date - start_date + 1):
        combined_returns[day] = all_returns[chosen_strat_idx[day], day]
    
    combined_cumulative = np.zeros(end_date - start_date + 1)
    combined_cumulative[0] = 1.0
    for d in range(1, end_date - start_date + 1):
        combined_cumulative[d] = combined_cumulative[d-1] * (1 + combined_returns[d])
    
    # Plot combined strategy vs individual strategies
    plt.figure(figsize=(12, 6))
    for s in range(5):
        plt.plot(range(start_date, end_date + 1), cum_returns[s], alpha=0.5, linestyle='--', label=strategy_names[s])
    plt.plot(range(start_date, end_date + 1), combined_cumulative, linewidth=2, label='Combined Strategy')
    plt.title('Combined Strategy vs Individual Strategies')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('combined_vs_individual.png')

    # Calculate correlation matrix of strategy returns
    strategy_returns_df = pd.DataFrame(all_returns.T, columns=strategy_names)
    correlation_matrix = strategy_returns_df.corr()

    # Generate heatmap of strategy returns correlation
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Strategy Returns Correlation Matrix')
    # Add text annotations
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                     ha='center', va='center', color='black')
    plt.xticks(range(len(strategy_names)), strategy_names, rotation=45)
    plt.yticks(range(len(strategy_names)), strategy_names)
    plt.tight_layout()
    plt.savefig('strategy_correlation.png')

    columns = [''] + [str(i) for i in range(20)]
    output_df_weights = pd.DataFrame(output_weights, columns=columns)
    results = backtester_without_TC(output_df_weights)
    df_performance = pd.DataFrame({'Net Returns': [results[0]], 'Sharpe Ratio': [results[1]]})
    
    # Print performance summary
    print("\n===== Task 2 Performance Summary =====")
    print(f"Combined Strategy Net Returns: {results[0]:.2f}%")
    print(f"Combined Strategy Sharpe Ratio: {results[1]:.2f}")
    
    # Save individual strategy performance for comparison
    individual_performances = []
    for i, strat in enumerate(strategies):
        perf = backtester_without_TC(strat)
        individual_performances.append(perf)
        print(f"{strategy_names[i]} Net Returns: {perf[0]:.2f}%, Sharpe: {perf[1]:.2f}")
        
    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    returns = [p[0] for p in individual_performances] + [results[0]]
    sharpes = [p[1] for p in individual_performances] + [results[1]]
    labels = strategy_names + ['Combined']
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, returns, width, label='Net Returns (%)')
    rects2 = ax.bar(x + width/2, sharpes, width, label='Sharpe Ratio')
    
    ax.set_title('Strategy Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('performance_comparison.png')
    
    df_performance.to_csv('task_2.csv', index=False)
    output_df_weights.to_csv('task2_weights.csv', index=False)
    with open('task2_model.pkl', 'wb') as f:
        pickle.dump(chosen_strat_idx, f)
    
    print("Task 2 completed. Results saved and visualizations generated.")
    return

def calculate_turnover(weights_df):
    weights_diff_df = abs(weights_df-weights_df.shift(1))
    turnover_symbols = weights_diff_df.sum()
    turnover = turnover_symbols.sum()
    return turnover

def backtester_with_TC(weights_df):
    #Update path for data here
    data = pd.read_csv('train_data.csv')

    weights_df = weights_df.fillna(0)

    turnover = calculate_turnover(weights_df)

    start_date = 3000
    end_date = 3499

    transaction_cost = (turnover * 0.01)

    df_returns = pd.DataFrame()

    for i in range(0,20):
        data_symbol = data[data['Symbol']==i]
        data_symbol = data_symbol['Close']
        data_symbol = data_symbol.reset_index(drop=True)   
        data_symbol = data_symbol/data_symbol.shift(1) - 1
        df_returns =  pd.concat([df_returns,data_symbol], axis=1, ignore_index=True)
    
    df_returns = df_returns.fillna(0)
    
    weights_df = weights_df.loc[start_date:end_date]    
    df_returns = df_returns.loc[start_date:end_date]

    df_returns = weights_df.mul(df_returns)

    initial_notional = 1
    notional = initial_notional

    returns = []

    for date in range(start_date,end_date+1):
        returns.append(df_returns.loc[date].values.sum())
        notional = notional * (1+returns[date-start_date])

    net_return = ((notional - transaction_cost - initial_notional)/initial_notional)*100
    sharpe_ratio = (pd.DataFrame(returns).mean().values[0] - (transaction_cost/(end_date-start_date+1)))/pd.DataFrame(returns).std().values[0]

    return [net_return, sharpe_ratio]



def task3():
    import matplotlib.pyplot as plt
    
    print("Starting Task 3...")
    strategies = [task1_Strategy1(), task1_Strategy2(), task1_Strategy3(), task1_Strategy4(), task1_Strategy5()]
    start_date, end_date = 3500, 3999
    data = pd.read_csv('cross_val_data.csv')
    all_returns = []
    all_weights = []
    strategy_names = ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4", "Strategy 5"]
    
    for strat in strategies:
        strat_ = strat.drop(strat.columns[0], axis=1)
        strat_.index = range(start_date, end_date+1)
        df_returns = pd.DataFrame()
        for i in range(20):
            data_symbol = data[data['Symbol'] == i]['Close'].reset_index(drop=True)
            data_symbol = data_symbol / data_symbol.shift(1) - 1
            df_returns = pd.concat([df_returns, data_symbol], axis=1, ignore_index=True)
        df_returns.index = range(start_date, end_date+1)
        df_returns = df_returns.fillna(0)
        strat_ = strat_.loc[start_date:end_date]
        df_returns = df_returns.loc[start_date:end_date]
        daily_returns = (strat_.values * df_returns.values).sum(axis=1)
        all_returns.append(daily_returns)
        all_weights.append(strat_.values)
    all_returns = np.array(all_returns)  # shape: (5, 500)
    all_weights = np.array(all_weights)  # shape: (5, 500, 20)

    # Calculate strategy turnovers - for visualization
    strategy_turnovers = np.zeros((5, end_date - start_date))
    for s in range(5):
        for d in range(1, end_date - start_date + 1):
            strategy_turnovers[s, d-1] = np.sum(np.abs(all_weights[s, d] - all_weights[s, d-1]))
    
    # Plot average turnover by strategy
    plt.figure(figsize=(10, 6))
    avg_turnovers = np.mean(strategy_turnovers, axis=1)
    plt.bar(strategy_names, avg_turnovers)
    plt.title('Average Daily Turnover by Strategy')
    plt.ylabel('Average Turnover')
    plt.grid(True, alpha=0.3)
    plt.savefig('task3_avg_turnover.png')
    
    # Calculate cumulative returns for each strategy (before TC)
    cum_returns_no_tc = np.zeros((5, end_date - start_date + 1))
    for s in range(5):
        cum_returns_no_tc[s, 0] = 1.0  # Start with $1
        for d in range(1, end_date - start_date + 1):
            cum_returns_no_tc[s, d] = cum_returns_no_tc[s, d-1] * (1 + all_returns[s, d])
    
    prev_weights = np.zeros(20)
    output_weights = []
    chosen_strat_idx = []
    net_returns_list = []
    daily_turnover = []
    daily_tc_cost = []
    
    # Run strategy selection with transaction cost consideration
    for day in range(end_date - start_date + 1):
        # Strategy selection logic
        if day < 10:
            idx = 0  # Default to first strategy for first 10 days
        else:
            # Use volatility as selection criterion
            recent_vol = all_returns[:, day-10:day].std(axis=1)
            idx = np.argmin(recent_vol)
        
        # Track selected strategy
        chosen_strat_idx.append(idx)
        
        # Calculate turnover and net return
        turnover = np.sum(np.abs(all_weights[idx, day] - prev_weights))
        tc_cost = 0.01 * turnover
        net_return = all_returns[idx, day] - tc_cost
        
        # Store metrics
        net_returns_list.append(net_return)
        daily_turnover.append(turnover)
        daily_tc_cost.append(tc_cost)
        
        # Update previous weights
        prev_weights = all_weights[idx, day].copy()
        weights_row = [start_date + day] + list(all_weights[idx, day])
        output_weights.append(weights_row)
    
    # Calculate cumulative returns with transaction costs
    cum_net_returns = np.zeros(end_date - start_date + 1)
    cum_net_returns[0] = 1.0
    for d in range(1, end_date - start_date + 1):
        cum_net_returns[d] = cum_net_returns[d-1] * (1 + net_returns_list[d-1])
    
    # Plot cumulative returns with TC vs without TC for chosen strategy
    plt.figure(figsize=(12, 6))
    plt.plot(range(start_date, end_date + 1), cum_net_returns, 'b-', linewidth=2, label='Combined with TC')
    
    # Create a theoretical combined without TC line
    combined_no_tc = np.zeros(end_date - start_date + 1)
    combined_no_tc[0] = 1.0
    for d in range(1, end_date - start_date + 1):
        combined_no_tc[d] = combined_no_tc[d-1] * (1 + all_returns[chosen_strat_idx[d-1], d-1])
    
    plt.plot(range(start_date, end_date + 1), combined_no_tc, 'g--', linewidth=1.5, label='Combined without TC')
    plt.title('Impact of Transaction Costs on Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('task3_tc_impact.png')
    
    # Plot strategy selection frequency
    strategy_counts = np.bincount(chosen_strat_idx, minlength=5)
    plt.figure(figsize=(10, 6))
    plt.bar(strategy_names, strategy_counts)
    plt.title('Strategy Selection Frequency (with TC consideration)')
    plt.ylabel('Number of Days Selected')
    plt.grid(True, alpha=0.3)
    plt.savefig('task3_strategy_selection.png')
    
    # Plot daily turnover
    plt.figure(figsize=(12, 6))
    plt.plot(range(start_date, end_date + 1), daily_turnover)
    plt.title('Daily Portfolio Turnover')
    plt.xlabel('Date')
    plt.ylabel('Turnover')
    plt.grid(True, alpha=0.3)
    plt.savefig('task3_daily_turnover.png')
    
    # Plot transaction cost over time
    plt.figure(figsize=(12, 6))
    plt.plot(range(start_date, end_date + 1), daily_tc_cost)
    plt.title('Daily Transaction Costs')
    plt.xlabel('Date')
    plt.ylabel('Cost (%)')
    plt.grid(True, alpha=0.3)
    plt.fill_between(range(start_date, end_date + 1), 0, daily_tc_cost, alpha=0.3)
    plt.savefig('task3_tc_costs.png')
    
    # Plot strategy selection over time
    plt.figure(figsize=(12, 6))
    plt.plot(range(start_date, end_date + 1), chosen_strat_idx)
    plt.yticks(range(5), strategy_names)
    plt.title('Strategy Selection Over Time (with TC)')
    plt.xlabel('Date')
    plt.ylabel('Selected Strategy')
    plt.grid(True, alpha=0.3)
    plt.savefig('task3_strategy_timeline.png')
    
    # Calculate proper compound returns
    notional = 1.0
    for ret in net_returns_list:
        notional *= (1 + ret)
    total_net_return = (notional - 1) * 100  # Convert to percentage
    
    sharpe = np.mean(net_returns_list) / (np.std(net_returns_list) + 1e-8)
    
    # Performance comparison - with and without TC
    plt.figure(figsize=(10, 6))
    
    # Calculate returns for each strategy with TC
    strategy_tc_returns = []
    strategy_tc_sharpes = []
    
    for s in range(5):
        prev_w = np.zeros(20)
        net_rets = []
        
        for d in range(end_date - start_date + 1):
            turnover_s = np.sum(np.abs(all_weights[s, d] - prev_w))
            net_ret_s = all_returns[s, d] - 0.01 * turnover_s
            net_rets.append(net_ret_s)
            prev_w = all_weights[s, d].copy()
        
        # Compound returns
        notional_s = 1.0
        for ret in net_rets:
            notional_s *= (1 + ret)
        
        strategy_tc_returns.append((notional_s - 1) * 100)
        strategy_tc_sharpes.append(np.mean(net_rets) / (np.std(net_rets) + 1e-8))
    
    # Create bar chart
    x = np.arange(len(strategy_names) + 1)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    returns_with_tc = strategy_tc_returns + [total_net_return]
    sharpes_with_tc = strategy_tc_sharpes + [sharpe]
    labels = strategy_names + ['Combined']
    
    rects1 = ax.bar(x - width/2, returns_with_tc, width, label='Net Returns with TC (%)')
    rects2 = ax.bar(x + width/2, sharpes_with_tc, width, label='Sharpe Ratio with TC')
    
    ax.set_title('Strategy Performance Comparison (with Transaction Costs)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('task3_performance_comparison.png')
    
    # Save results
    columns = [''] + [str(i) for i in range(20)]
    output_df_weights = pd.DataFrame(output_weights, columns=columns)
    df_performance = pd.DataFrame({'Net Returns': [total_net_return], 'Sharpe Ratio': [sharpe]})
    df_performance.to_csv('task_3.csv', index=False)
    output_df_weights.to_csv('task3_weights.csv', index=False)
    with open('task3_model.pkl', 'wb') as f:
        pickle.dump(chosen_strat_idx, f)
    
    # Cross-check with backtester
    test_result = backtester_with_TC(output_df_weights)
    print("\n===== Task 3 Performance Summary =====")
    print(f"Combined Strategy Net Returns (with TC): {total_net_return:.2f}%")
    print(f"Combined Strategy Sharpe Ratio: {sharpe:.2f}")
    print(f"Backtester TC result: {test_result}")
    print("Task 3 completed. Results saved and visualizations generated.")
    return



if __name__ == '__main__':
    task1()
    task2()
    task3()