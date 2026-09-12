# Stock Price Movement Predictor

This repository contains a leak-free, honestly-evaluated next-day stock direction classifier for a liquid equity (SPY).
It uses daily OHLCV history downloaded via `yfinance` to predict whether the next day's close will be strictly greater than today's close (Up) or not (Down/Flat).

## Project Structure
- `notebook.ipynb`: The Jupyter Notebook containing the full pipeline, from data fetching and target construction to model training and evaluation. It includes explicit checks for zero leakage.
- `README.md`: Project documentation and interpretation of results.
- `requirements.txt`: Python dependencies.
- `data/`: Directory where the raw stock data is cached.

## Requirements and Execution
To run the notebook:
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Approach

1. **Target Construction with Zero Leakage**: 
   The target is constructed as `1 if close.shift(-1) > close else 0`. A specific leakage verification check is printed in the notebook to confirm that the `target` column precisely maps to the comparison between the current day's close and the *next* day's close, ensuring no forward-looking data contaminates the current row's features.

2. **Feature Sets**:
   - **Raw Features**: OHLCV data directly from Yahoo Finance, alongside a simple 1-day lagged return.
   - **Engineered Features**: Using the `ta` library, we compute the Relative Strength Index (RSI), MACD, and 20-day rolling volatility. NaN values from the warmup period are rigorously dropped before splitting to preserve the temporal integrity.

3. **Chronological Split**:
   To emulate real-world trading conditions, the data is split chronologically into ~80% training and ~20% test data. Data is **never shuffled**. A `StandardScaler` is fitted *only* on the training set and is then used to transform both train and test sets to prevent data leakage from the future into the past.

4. **Baselines**:
   - **Persistence**: Predicts that tomorrow's direction will be the same as today's realized direction.
   - **Majority Class**: Always predicts the majority class label observed in the training split.

## Four-Way Comparison Results

| Model                     |   Accuracy |   Precision |   Recall |   F1 Score |
|:--------------------------|-----------:|------------:|---------:|-----------:|
| Persistence               |   0.506944 |    0.551208 | 0.550575 |   0.550891 |
| Majority Class            |   0.549242 |    0.549242 | 1.000000 |   0.709046 |
| Raw Features Model        |   0.549242 |    0.549618 | 0.993103 |   0.707617 |
| Engineered Features Model |   0.561869 |    0.560357 | 0.939080 |   0.701890 |

## Honest Interpretation of Results

- **Performance vs Baselines**: Running on 31 years of data (1993-2024), the Engineered Features Model achieved a 56.1% accuracy on the test set, beating the Majority Class (54.9%) and Persistence (50.6%) baselines. The Raw Features Model (54.9%) again defaulted to acting as a majority-class predictor (indicated by its 99% recall and baseline-equivalent accuracy), failing to extract signal.
- **Signal vs. Noise**: With the expanded test set size of roughly 1,590 trading days (~20% of 7,954 total days), the ~1.2% absolute outperformance over the majority class equates to around 19 extra correct predictions over the multi-year test window. While slightly more robust due to the larger sample size, this is still a thin margin for a real-world trading system and might be swallowed entirely by transaction costs.
- **Class Balance & Baseline Difficulty**: The overall target distribution remains imbalanced (~54% Up, 46% Down/Flat) over the 31-year history. This built-in bias solidifies the Majority Class as an inherently difficult baseline to beat in directional forecasting for major stock indices.

## Real-World Limitations

- **No Transaction Costs or Slippage**: This model assumes we can execute at exactly the closing price of the day we make our prediction. In reality, trading at the close or opening bell incurs slippage, and brokerage fees/transaction costs would eat into any potential profit.
- **Not Real-Time**: The model runs on end-of-day data, meaning predictions are generated for the next day, precluding any intraday opportunities.
- **Market Efficiency**: Financial markets are highly efficient. Any marginal edge discovered using simple technical indicators is often already arbitraged away by larger institutional players, and achieving a 52-54% accuracy without transaction costs does not guarantee a profitable trading strategy.
