# stock-options

A Rust library for calculating option pricing Greeks using the Black-Scholes and Bjerksund-Stensland models.

## Features

- **Black-Scholes Model**: Calculate Greeks for European-style options
- **Bjerksund-Stensland Model**: Calculate Greeks for American-style options with early exercise
- **Complete Greeks Suite**: Delta, Gamma, Theta, Vega, and Rho
- **Batch Calculation**: Calculate all Greeks at once with `all_greeks()`
- **Type-Safe**: Strongly typed `OptionType` enum prevents errors
- **Zero Dependencies on NumPy/SciPy**: Pure Rust implementation using `statrs`

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
stock-options = { git = "https://github.com/alexjokela/stock-options" }
```

## Usage

### Basic Example

```rust
use stock_options::{black_scholes, bjerksund_stensland, OptionType};

fn main() {
    // Parameters
    let s = 100.0;      // Current stock price
    let k = 105.0;      // Strike price
    let t = 0.5;        // Time to expiration (years)
    let r = 0.05;       // Risk-free interest rate
    let sigma = 0.2;    // Volatility
    let q = 0.01;       // Dividend yield

    // Calculate Black-Scholes delta for a call option
    let delta = black_scholes::delta(s, k, t, r, sigma, q, OptionType::Call).unwrap();
    println!("Black-Scholes Call Delta: {:.4}", delta);

    // Calculate Black-Scholes delta for a put option
    let put_delta = black_scholes::delta(s, k, t, r, sigma, q, OptionType::Put).unwrap();
    println!("Black-Scholes Put Delta: {:.4}", put_delta);
}
```

### Calculate All Greeks at Once

```rust
use stock_options::{black_scholes, OptionType};

fn main() {
    let greeks = black_scholes::all_greeks(
        100.0,              // Stock price
        105.0,              // Strike price
        0.5,                // Time to expiration
        0.05,               // Risk-free rate
        0.2,                // Volatility
        0.01,               // Dividend yield
        OptionType::Call
    ).unwrap();

    println!("Delta: {:.4}", greeks.delta);
    println!("Gamma: {:.4}", greeks.gamma);
    println!("Theta: {:.4} (per day)", greeks.theta);
    println!("Vega:  {:.4}", greeks.vega);
    println!("Rho:   {:.4} (per 1% rate change)", greeks.rho);
}
```

### American Options with Bjerksund-Stensland

```rust
use stock_options::{bjerksund_stensland, OptionType};

fn main() {
    // American options can be exercised early, which affects their pricing
    let american_greeks = bjerksund_stensland::all_greeks(
        100.0,              // Stock price
        95.0,               // Strike price
        1.0,                // Time to expiration
        0.05,               // Risk-free rate
        0.25,               // Volatility
        0.02,               // Dividend yield
        OptionType::Put
    ).unwrap();

    println!("American Put Greeks:");
    println!("  Delta: {:.4}", american_greeks.delta);
    println!("  Gamma: {:.4}", american_greeks.gamma);
    println!("  Theta: {:.4}", american_greeks.theta);
    println!("  Vega:  {:.4}", american_greeks.vega);
    println!("  Rho:   {:.4}", american_greeks.rho);
}
```

## API Reference

### Types

#### `OptionType`
```rust
pub enum OptionType {
    Call,
    Put,
}
```

#### `Greeks`
```rust
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,  // Per day
    pub vega: f64,
    pub rho: f64,    // Per 1% change in interest rate
}
```

### Black-Scholes Functions (European Options)

All functions take the following parameters:
- `s: f64` - Current price of the underlying asset
- `k: f64` - Strike price of the option
- `t: f64` - Time to expiration in years
- `r: f64` - Risk-free interest rate (e.g., 0.05 for 5%)
- `sigma: f64` - Volatility of the underlying asset (e.g., 0.2 for 20%)
- `q: f64` - Dividend yield (e.g., 0.01 for 1%)
- `option_type: OptionType` - Call or Put (where applicable)

| Function | Description |
|----------|-------------|
| `black_scholes::delta()` | Sensitivity to underlying price |
| `black_scholes::gamma()` | Rate of change of delta |
| `black_scholes::theta()` | Time decay (per day) |
| `black_scholes::vega()` | Sensitivity to volatility |
| `black_scholes::rho()` | Sensitivity to interest rate |
| `black_scholes::all_greeks()` | Calculate all Greeks at once |

### Bjerksund-Stensland Functions (American Options)

Same parameters as Black-Scholes functions.

| Function | Description |
|----------|-------------|
| `bjerksund_stensland::delta()` | Sensitivity to underlying price |
| `bjerksund_stensland::gamma()` | Rate of change of delta |
| `bjerksund_stensland::theta()` | Time decay (per day) |
| `bjerksund_stensland::vega()` | Sensitivity to volatility |
| `bjerksund_stensland::rho()` | Sensitivity to interest rate |
| `bjerksund_stensland::all_greeks()` | Calculate all Greeks at once |

## The Greeks Explained

| Greek | Symbol | Description |
|-------|--------|-------------|
| **Delta** | Δ | Measures how much the option price changes for a $1 change in the underlying asset price. Call deltas range from 0 to 1; put deltas range from -1 to 0. |
| **Gamma** | Γ | Measures the rate of change of delta. High gamma means delta is very sensitive to price changes. Same for calls and puts. |
| **Theta** | Θ | Measures time decay - how much value the option loses per day. Usually negative (options lose value over time). |
| **Vega** | ν | Measures sensitivity to volatility. A vega of 0.10 means a 1% increase in volatility increases the option price by $0.10. |
| **Rho** | ρ | Measures sensitivity to interest rate changes. Calls have positive rho; puts have negative rho. |

## Model Comparison

| Feature | Black-Scholes | Bjerksund-Stensland |
|---------|---------------|---------------------|
| Option Style | European | American |
| Early Exercise | No | Yes |
| Complexity | Lower | Higher |
| Use Case | European options, index options | Stock options, ETF options |

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Bjerksund, P., & Stensland, G. (2002). Closed Form Valuation of American Options. *Discussion paper 2002/09*, Norwegian School of Economics.
