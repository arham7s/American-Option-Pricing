import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/american_option_mplconfig")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def payoff(stock_price, strike, option_type):
    if option_type == "call":
        return max(stock_price - strike, 0.0)
    if option_type == "put":
        return max(strike - stock_price, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(spot, strike, maturity, rate, volatility, option_type, dividend_yield=0.0):
    if maturity <= 0:
        return payoff(spot, strike, option_type)
    if volatility <= 0:
        forward_intrinsic = spot * math.exp(-dividend_yield * maturity) - strike * math.exp(-rate * maturity)
        if option_type == "call":
            return max(forward_intrinsic, 0.0)
        return max(-forward_intrinsic, 0.0)

    sqrt_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)

    if option_type == "call":
        return discounted_spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    return discounted_strike * norm_cdf(-d2) - discounted_spot * norm_cdf(-d1)


def american_option_binomial(
    spot,
    strike,
    maturity,
    rate,
    volatility,
    steps,
    option_type,
    dividend_yield=0.0,
    return_boundary=False,
):
    dt = maturity / steps
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((rate - dividend_yield) * dt)
    p = (growth - d) / (u - d)
    discount = math.exp(-rate * dt)

    if not 0.0 <= p <= 1.0:
        raise ValueError("Risk-neutral probability is outside [0, 1]. Increase steps.")

    stock_prices = np.array([spot * (u**j) * (d ** (steps - j)) for j in range(steps + 1)], dtype=float)
    option_values = np.array([payoff(price, strike, option_type) for price in stock_prices], dtype=float)

    step_times = np.linspace(0.0, maturity, steps + 1)
    exercise_boundary = np.full(steps + 1, np.nan)

    for i in range(steps - 1, -1, -1):
        stock_prices = stock_prices[:-1] / d
        continuation = discount * (p * option_values[1:] + (1.0 - p) * option_values[:-1])
        intrinsic = np.array([payoff(price, strike, option_type) for price in stock_prices], dtype=float)
        exercise = intrinsic > continuation
        option_values = np.maximum(intrinsic, continuation)

        if return_boundary and np.any(exercise):
            exercised_prices = stock_prices[exercise]
            if option_type == "put":
                exercise_boundary[i] = np.max(exercised_prices)
            else:
                exercise_boundary[i] = np.min(exercised_prices)

    result = {
        "price": float(option_values[0]),
        "method": "Binomial",
        "steps": steps,
    }
    if return_boundary:
        result["boundary_times"] = step_times
        result["exercise_boundary"] = exercise_boundary
    return result


def american_option_trinomial(
    spot,
    strike,
    maturity,
    rate,
    volatility,
    steps,
    option_type,
    dividend_yield=0.0,
    return_boundary=False,
):
    dt = maturity / steps
    up = math.exp(volatility * math.sqrt(2.0 * dt))
    discount = math.exp(-rate * dt)

    exp_half_drift = math.exp((rate - dividend_yield) * dt / 2.0)
    exp_up_half = math.exp(volatility * math.sqrt(dt / 2.0))
    exp_down_half = math.exp(-volatility * math.sqrt(dt / 2.0))
    denominator = exp_up_half - exp_down_half

    p_up = ((exp_half_drift - exp_down_half) / denominator) ** 2
    p_down = ((exp_up_half - exp_half_drift) / denominator) ** 2
    p_mid = 1.0 - p_up - p_down

    probabilities = (p_down, p_mid, p_up)
    if any(prob < -1e-12 or prob > 1.0 + 1e-12 for prob in probabilities):
        raise ValueError("Trinomial probabilities are unstable for this step size. Increase steps.")

    p_down = max(0.0, p_down)
    p_mid = max(0.0, p_mid)
    p_up = max(0.0, p_up)
    prob_sum = p_down + p_mid + p_up
    p_down, p_mid, p_up = p_down / prob_sum, p_mid / prob_sum, p_up / prob_sum

    grid = np.arange(-steps, steps + 1)
    stock_prices = spot * (up ** grid)
    option_values = np.array([payoff(price, strike, option_type) for price in stock_prices], dtype=float)

    step_times = np.linspace(0.0, maturity, steps + 1)
    exercise_boundary = np.full(steps + 1, np.nan)

    for i in range(steps - 1, -1, -1):
        current_grid = np.arange(-i, i + 1)
        current_prices = spot * (up ** current_grid)
        continuation = discount * (
            p_up * option_values[current_grid + steps + 1]
            + p_mid * option_values[current_grid + steps]
            + p_down * option_values[current_grid + steps - 1]
        )
        intrinsic = np.array([payoff(price, strike, option_type) for price in current_prices], dtype=float)
        exercise = intrinsic > continuation

        updated_values = np.maximum(intrinsic, continuation)
        option_values = np.zeros(2 * steps + 1, dtype=float)
        option_values[current_grid + steps] = updated_values

        if return_boundary and np.any(exercise):
            exercised_prices = current_prices[exercise]
            if option_type == "put":
                exercise_boundary[i] = np.max(exercised_prices)
            else:
                exercise_boundary[i] = np.min(exercised_prices)

    result = {
        "price": float(option_values[steps]),
        "method": "Trinomial",
        "steps": steps,
    }
    if return_boundary:
        result["boundary_times"] = step_times
        result["exercise_boundary"] = exercise_boundary
    return result


def run_convergence_study(params, option_type, step_grid):
    rows = []
    for steps in step_grid:
        start = time.perf_counter()
        binomial = american_option_binomial(option_type=option_type, steps=steps, **params)
        binomial_time = time.perf_counter() - start

        start = time.perf_counter()
        trinomial = american_option_trinomial(option_type=option_type, steps=steps, **params)
        trinomial_time = time.perf_counter() - start

        rows.append(
            {
                "steps": steps,
                "binomial_price": binomial["price"],
                "trinomial_price": trinomial["price"],
                "binomial_runtime_sec": binomial_time,
                "trinomial_runtime_sec": trinomial_time,
            }
        )

    study = pd.DataFrame(rows)
    study["absolute_price_gap"] = (study["binomial_price"] - study["trinomial_price"]).abs()
    return study


def build_summary_table(params, put_study, call_study, reference_steps_binomial=1000, reference_steps_trinomial=500):
    european_put = black_scholes_price(option_type="put", **params)
    european_call = black_scholes_price(option_type="call", **params)

    final_put_binomial = put_study.iloc[-1]["binomial_price"]
    final_put_trinomial = put_study.iloc[-1]["trinomial_price"]
    final_call_binomial = call_study.iloc[-1]["binomial_price"]
    final_call_trinomial = call_study.iloc[-1]["trinomial_price"]

    refined_put_binomial = american_option_binomial(
        option_type="put",
        steps=reference_steps_binomial,
        **params,
    )["price"]
    refined_put_trinomial = american_option_trinomial(
        option_type="put",
        steps=reference_steps_trinomial,
        **params,
    )["price"]

    summary = pd.DataFrame(
        [
            {
                "metric": "European put (Black-Scholes)",
                "value": european_put,
            },
            {
                "metric": "American put (Binomial, final grid)",
                "value": final_put_binomial,
            },
            {
                "metric": "American put (Trinomial, final grid)",
                "value": final_put_trinomial,
            },
            {
                "metric": "American put premium vs European (Binomial)",
                "value": final_put_binomial - european_put,
            },
            {
                "metric": "American put premium vs European (Trinomial)",
                "value": final_put_trinomial - european_put,
            },
            {
                "metric": "Refined American put (Binomial)",
                "value": refined_put_binomial,
            },
            {
                "metric": "Refined American put (Trinomial)",
                "value": refined_put_trinomial,
            },
            {
                "metric": "European call (Black-Scholes)",
                "value": european_call,
            },
            {
                "metric": "American call (Binomial, final grid)",
                "value": final_call_binomial,
            },
            {
                "metric": "American call (Trinomial, final grid)",
                "value": final_call_trinomial,
            },
            {
                "metric": "American call minus European call (Binomial)",
                "value": final_call_binomial - european_call,
            },
            {
                "metric": "American call minus European call (Trinomial)",
                "value": final_call_trinomial - european_call,
            },
        ]
    )
    return summary


def plot_convergence(study, option_label, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(study["steps"], study["binomial_price"], marker="o", linewidth=2, label="Binomial")
    plt.plot(study["steps"], study["trinomial_price"], marker="s", linewidth=2, label="Trinomial")
    plt.xlabel("Number of Time Steps")
    plt.ylabel("Option Price")
    plt.title(f"American {option_label} Price Convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_runtime(put_study, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(
        put_study["steps"],
        put_study["binomial_runtime_sec"],
        marker="o",
        linewidth=2,
        label="Binomial runtime",
    )
    plt.plot(
        put_study["steps"],
        put_study["trinomial_runtime_sec"],
        marker="s",
        linewidth=2,
        label="Trinomial runtime",
    )
    plt.xlabel("Number of Time Steps")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison for American Put Pricing")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_boundary(binomial_result, trinomial_result, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(
        binomial_result["boundary_times"],
        binomial_result["exercise_boundary"],
        linewidth=2,
        label="Binomial boundary",
    )
    plt.plot(
        trinomial_result["boundary_times"],
        trinomial_result["exercise_boundary"],
        linewidth=2,
        label="Trinomial boundary",
    )
    plt.xlabel("Time")
    plt.ylabel("Critical Stock Price")
    plt.title("Early Exercise Boundary for the American Put")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_project_summary(params, put_study, call_study, summary_table):
    print("\nAMERICAN OPTION PRICING PROJECT")
    print("-" * 40)
    print("Model parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    print("\nFinal convergence values:")
    print(put_study.tail(1).to_string(index=False))
    print(call_study.tail(1).to_string(index=False))

    print("\nSummary metrics:")
    print(summary_table.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    print("\nInterpretation:")
    print("1. The American put should price above the European put because early exercise can be optimal.")
    print("2. The American call on a non-dividend-paying stock should be very close to the European call.")
    print("3. The trinomial tree often converges smoothly, while the binomial tree is simpler to implement.")


def main():
    output_dir = Path("american_option_results")
    output_dir.mkdir(exist_ok=True)

    params = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.06,
        "volatility": 0.20,
        "dividend_yield": 0.0,
    }

    step_grid = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300]

    put_study = run_convergence_study(params, option_type="put", step_grid=step_grid)
    call_study = run_convergence_study(params, option_type="call", step_grid=step_grid)
    summary_table = build_summary_table(params, put_study, call_study)

    boundary_steps = 200
    put_boundary_binomial = american_option_binomial(
        option_type="put",
        steps=boundary_steps,
        return_boundary=True,
        **params,
    )
    put_boundary_trinomial = american_option_trinomial(
        option_type="put",
        steps=boundary_steps,
        return_boundary=True,
        **params,
    )

    put_study.to_csv(output_dir / "put_convergence_study.csv", index=False)
    call_study.to_csv(output_dir / "call_convergence_study.csv", index=False)
    summary_table.to_csv(output_dir / "summary_metrics.csv", index=False)

    plot_convergence(put_study, "Put", output_dir / "american_put_convergence.png")
    plot_convergence(call_study, "Call", output_dir / "american_call_convergence.png")
    plot_runtime(put_study, output_dir / "runtime_comparison.png")
    plot_boundary(
        put_boundary_binomial,
        put_boundary_trinomial,
        output_dir / "early_exercise_boundary_put.png",
    )

    print_project_summary(params, put_study, call_study, summary_table)

    print(f"\nSaved CSV files and plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
