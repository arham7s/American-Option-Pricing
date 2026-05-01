# American Option Pricing Using Binomial and Trinomial Trees

This project studies how American options can be priced using two lattice-based numerical methods:

- the **Binomial Tree**
- the **Trinomial Tree**

The main goal is to compare these methods, understand the effect of the **early exercise** feature in American options, and present the results through:

- a beginner-friendly website
- a dataset-driven option tester
- a Python script that generates model outputs and graphs

## Live Website

- Main page: [https://docs-brown-beta.vercel.app](https://docs-brown-beta.vercel.app)
- Option tester page: [https://docs-brown-beta.vercel.app/option-tester.html](https://docs-brown-beta.vercel.app/option-tester.html)

## Project Files

### Frontend

- Main teaching site: `docs/index.html`
- Dataset option tester: `docs/option-tester.html`
- Corrected dataset: `docs/us_options_yfinance_dataset_corrected.csv`
- Bundled dataset fallback: `docs/us_options_dataset_data.js`

### Python

- Pricing script: `american_option_pricing_project.py`

### Generated Output

Saved in `american_option_results/`:

- `put_convergence_study.csv`
- `call_convergence_study.csv`
- `summary_metrics.csv`
- `american_put_convergence.png`
- `american_call_convergence.png`
- `runtime_comparison.png`
- `early_exercise_boundary_put.png`

## What the Project Covers

### 1. American Option Pricing

American options can be exercised **any time before expiry**, unlike European options, which can only be exercised at maturity.

Because of this, pricing an American option requires checking at each step whether:

- exercising immediately is better, or
- continuing to hold the option is better

This is why tree-based methods are useful.

### 2. Binomial Tree Method

The binomial tree assumes that at each time step the stock can move:

- **up**
- **down**

The method:

1. builds the stock-price tree forward
2. computes option payoffs at maturity
3. works backward to the present
4. compares **intrinsic value** and **continuation value** at each node

### 3. Trinomial Tree Method

The trinomial tree extends the idea by allowing three moves at each step:

- **up**
- **middle**
- **down**

This often gives smoother numerical convergence than the binomial tree while using the same backward-induction logic.

### 4. Comparison and Analysis

The project compares:

- American binomial price
- American trinomial price
- European Black-Scholes benchmark
- convergence as the number of steps increases
- runtime of each method
- early exercise boundary for the American put

## Dataset-Based Option Tester

The second webpage, `docs/option-tester.html`, uses the corrected CSV dataset to test real option contracts.

It allows the user to:

- inspect dataset coverage by ticker, expiry, and option type
- see ticker-level distribution charts
- choose a contract from the dataset
- compare observed market price with model prices
- switch volatility source between:
  - implied volatility
  - historical volatility
  - manual volatility

The tester uses:

- `mid_price` when available
- otherwise `market_price`
- otherwise `lastPrice`

## Python Script

The Python file `american_option_pricing_project.py`:

- prices American calls and puts with binomial and trinomial trees
- computes European Black-Scholes benchmark values
- performs convergence studies
- compares runtime
- plots the early exercise boundary
- saves CSV and PNG outputs

## Default Model Inputs in the Python Script

The base example in the script uses:

- Spot price = `100`
- Strike price = `100`
- Maturity = `1.0` year
- Risk-free rate = `0.06`
- Volatility = `0.20`
- Dividend yield = `0.00`

Step grid used for comparison:

- `25, 50, 75, 100, 125, 150, 175, 200, 250, 300`

## How to Run the Python Script

You can run it locally or in Google Colab.

### Local / Colab steps

1. Open `american_option_pricing_project.py`
2. Install required libraries if needed:

```python
!pip install numpy pandas matplotlib
```

3. Run the script

It will create the folder:

```text
american_option_results/
```

with the graphs and CSV files.

## How to Use the Frontend

### Main page

Open:

```text
docs/index.html
```

This page explains:

- American vs European options
- binomial and trinomial tree formulas
- beginner-friendly intuition
- interactive tree walkthroughs

### Dataset tester page

Open:

```text
docs/option-tester.html
```

This page:

- loads the corrected dataset
- summarizes the dataset visually
- lets you test specific option contracts

## Deployment

The project is currently hosted on Vercel:

- [https://docs-brown-beta.vercel.app](https://docs-brown-beta.vercel.app)
- [https://docs-brown-beta.vercel.app/option-tester.html](https://docs-brown-beta.vercel.app/option-tester.html)

## Conclusion

This project demonstrates that American option pricing is more complex than European option pricing because of the possibility of early exercise. Binomial and trinomial trees provide practical numerical solutions, and the project shows both the theory and the applied side through code, visual explanation, and real dataset testing.
