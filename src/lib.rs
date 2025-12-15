//! # stock-options
//!
//! A Rust library for calculating option pricing Greeks using the Black-Scholes
//! and Bjerksund-Stensland models.
//!
//! ## Features
//!
//! - **Black-Scholes Greeks**: Delta, Gamma, Theta, Vega, and Rho for European options
//! - **Bjerksund-Stensland Greeks**: Delta, Gamma, Theta, Vega, and Rho for American options
//!
//! ## Example
//!
//! ```rust
//! use stock_options::{black_scholes, bjerksund_stensland, OptionType};
//!
//! // Calculate Black-Scholes delta for a call option
//! let delta = black_scholes::delta(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
//! println!("Delta: {}", delta);
//!
//! // Calculate all Greeks at once
//! let greeks = black_scholes::all_greeks(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
//! println!("Delta: {}, Gamma: {}, Theta: {}, Vega: {}, Rho: {}",
//!     greeks.delta, greeks.gamma, greeks.theta, greeks.vega, greeks.rho);
//! ```

use std::f64::consts::E;
use statrs::distribution::Normal;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Continuous;

/// Option type: Call or Put
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

/// Error type for option calculations
#[derive(Debug, Clone, PartialEq)]
pub enum OptionError {
    InvalidOptionType,
    InvalidParameters(String),
}

impl std::fmt::Display for OptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionError::InvalidOptionType => write!(f, "Invalid option type"),
            OptionError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
        }
    }
}

impl std::error::Error for OptionError {}

/// Container for all option Greeks
#[derive(Debug, Clone, Copy)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Calculate the d1 parameter for the Black-Scholes model.
fn calculate_d1(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> f64 {
    (f64::ln(s / k) + (r - q + 0.5 * sigma.powi(2)) * t) / (sigma * f64::sqrt(t))
}

/// Black-Scholes model for European options
pub mod black_scholes {
    use super::*;

    /// Calculate the delta (sensitivity to the underlying price) of a European option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// The delta of the European option
    pub fn delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: OptionType) -> Result<f64, OptionError> {
        let d1 = calculate_d1(s, k, t, r, sigma, q);
        let norm = Normal::new(0.0, 1.0).unwrap();

        let delta = match option_type {
            OptionType::Call => f64::exp(-q * t) * norm.cdf(d1),
            OptionType::Put => -f64::exp(-q * t) * norm.cdf(-d1),
        };

        Ok(delta)
    }

    /// Calculate the gamma (second-order sensitivity to the underlying price) of a European option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    ///
    /// # Returns
    ///
    /// The gamma of the European option (same for calls and puts)
    pub fn gamma(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> Result<f64, OptionError> {
        let d1 = calculate_d1(s, k, t, r, sigma, q);
        let norm_dist = Normal::new(0.0, 1.0).unwrap();
        let gamma = norm_dist.pdf(d1) * E.powf(-q * t) / (s * sigma * f64::sqrt(t));

        Ok(gamma)
    }

    /// Calculate the theta (sensitivity to time to expiration) of a European option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// The theta of the European option (per day)
    pub fn theta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: OptionType) -> Result<f64, OptionError> {
        let d1 = calculate_d1(s, k, t, r, sigma, q);
        let d2 = d1 - sigma * f64::sqrt(t);
        let norm = Normal::new(0.0, 1.0).unwrap();

        let theta = match option_type {
            OptionType::Call => {
                - (s * sigma * f64::exp(-q * t) * norm.pdf(d1)) / (2.0 * f64::sqrt(t))
                - r * k * f64::exp(-r * t) * norm.cdf(d2)
                + q * s * f64::exp(-q * t) * norm.cdf(d1)
            },
            OptionType::Put => {
                - (s * sigma * f64::exp(-q * t) * norm.pdf(d1)) / (2.0 * f64::sqrt(t))
                + r * k * f64::exp(-r * t) * norm.cdf(-d2)
                - q * s * f64::exp(-q * t) * norm.cdf(-d1)
            },
        };

        // Convert to per-day theta
        Ok(theta / 365.0)
    }

    /// Calculate the vega (sensitivity to volatility) of a European option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    ///
    /// # Returns
    ///
    /// The vega of the European option (same for calls and puts)
    pub fn vega(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> Result<f64, OptionError> {
        let d1 = calculate_d1(s, k, t, r, sigma, q);
        let norm = Normal::new(0.0, 1.0).unwrap();

        let vega = s * f64::exp(-q * t) * norm.pdf(d1) * f64::sqrt(t);

        Ok(vega)
    }

    /// Calculate the rho (sensitivity to interest rate) of a European option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// The rho of the European option (per 1% change in interest rate)
    pub fn rho(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: OptionType) -> Result<f64, OptionError> {
        let d1 = calculate_d1(s, k, t, r, sigma, q);
        let d2 = d1 - sigma * f64::sqrt(t);
        let norm = Normal::new(0.0, 1.0).unwrap();

        let rho = match option_type {
            OptionType::Call => k * t * f64::exp(-r * t) * norm.cdf(d2),
            OptionType::Put => -k * t * f64::exp(-r * t) * norm.cdf(-d2),
        };

        Ok(rho / 100.0)
    }

    /// Calculate all Greeks for a European option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// A `Greeks` struct containing delta, gamma, theta, vega, and rho
    pub fn all_greeks(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: OptionType) -> Result<Greeks, OptionError> {
        Ok(Greeks {
            delta: delta(s, k, t, r, sigma, q, option_type)?,
            gamma: gamma(s, k, t, r, sigma, q)?,
            theta: theta(s, k, t, r, sigma, q, option_type)?,
            vega: vega(s, k, t, r, sigma, q)?,
            rho: rho(s, k, t, r, sigma, q, option_type)?,
        })
    }
}

/// Bjerksund-Stensland model for American options
pub mod bjerksund_stensland {
    use super::*;

    /// Calculate the delta (sensitivity to the underlying price) of an American option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// The delta of the American option
    pub fn delta(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        q: f64,
        option_type: OptionType,
    ) -> Result<f64, OptionError> {
        let epsilon = 0.00001;
        let t_sqrt = t.sqrt();
        let b = q;
        let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
        let b_infinity = beta / (beta - 1.0) * k;
        let b_zero = match option_type {
            OptionType::Call => k.max(r / (r - b) * k),
            OptionType::Put => k.min(r / (r - b) * k),
        };

        let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
        let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
        let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
        let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
        let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

        let d1 = (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
        let d2 = (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
        let d3 = ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);
        let d4 = ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);

        let n = Normal::new(0.0, 1.0).unwrap();
        let n_d1 = n.cdf(d1);
        let n_d2 = n.cdf(d2);
        let n_d3 = n.cdf(d3);
        let n_d4 = n.cdf(d4);

        let (alpha, beta) = match option_type {
            OptionType::Call => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
            OptionType::Put => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        };

        let kappa = if b >= r || b <= epsilon {
            0.0
        } else {
            2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
        };

        let delta = match option_type {
            OptionType::Call => n_d1 - (s / b_t_infinity).powf(x) * (n_d2 - kappa),
            OptionType::Put => -n_d3 + (s / b_t_zero).powf(-x) * (n_d4 + kappa),
        };

        Ok(delta)
    }

    /// Calculate the gamma (second-order sensitivity to the underlying price) of an American option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// The gamma of the American option
    pub fn gamma(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        q: f64,
        option_type: OptionType,
    ) -> Result<f64, OptionError> {
        let epsilon = 0.00001;
        let t_sqrt = t.sqrt();
        let b = q;

        let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
        let b_infinity = beta / (beta - 1.0) * k;
        let b_zero = match option_type {
            OptionType::Call => k.max(r / (r - b) * k),
            OptionType::Put => k.min(r / (r - b) * k),
        };

        let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
        let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
        let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
        let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
        let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

        let d1 = (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
        let d2 = (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
        let d3 = ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);
        let d4 = ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);

        let n = Normal::new(0.0, 1.0).unwrap();
        let n_d2 = n.cdf(d2);
        let n_d4 = n.cdf(d4);
        let n_prime_d1 = n.pdf(d1);
        let n_prime_d2 = n.pdf(d2);
        let n_prime_d3 = n.pdf(d3);
        let n_prime_d4 = n.pdf(d4);

        let (alpha, beta) = match option_type {
            OptionType::Call => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
            OptionType::Put => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        };

        let kappa = if b >= r || b <= epsilon {
            0.0
        } else {
            2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
        };

        let kappa_prime = if b >= r || b <= epsilon {
            0.0
        } else {
            2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.pdf(y) * (-alpha / (sigma * t_sqrt)) - (s / b_t_infinity).powf(beta) * n.pdf(y - 2.0 * alpha / (sigma * t_sqrt)) * (-alpha / (sigma * t_sqrt)) * (beta - 1.0))
        };

        let gamma = match option_type {
            OptionType::Call => {
                n_prime_d1 / (s * sigma * t_sqrt) - x * (s / b_t_infinity).powf(x - 1.0) * (n_d2 - kappa) / b_t_infinity - (s / b_t_infinity).powf(x) * (n_prime_d2 * d2 / (sigma * t_sqrt) - kappa_prime) / s
            },
            OptionType::Put => {
                n_prime_d3 / (s * sigma * t_sqrt) + x * (s / b_t_zero).powf(-x - 1.0) * (n_d4 + kappa) / b_t_zero + (s / b_t_zero).powf(-x) * (-n_prime_d4 * d4 / (sigma * t_sqrt) + kappa_prime) / s
            },
        };

        Ok(gamma)
    }

    /// Calculate the theta (sensitivity to time to expiration) of an American option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// The theta of the American option (per day)
    pub fn theta(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        q: f64,
        option_type: OptionType,
    ) -> Result<f64, OptionError> {
        let epsilon = 0.00001;
        let t_sqrt = t.sqrt();
        let b = q;
        let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
        let b_infinity = beta / (beta - 1.0) * k;
        let b_zero = match option_type {
            OptionType::Call => k.max(r / (r - b) * k),
            OptionType::Put => k.min(r / (r - b) * k),
        };

        let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
        let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
        let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
        let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
        let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

        let d1 = (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
        let d2 = (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
        let d3 = ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);
        let d4 = ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);

        let n = Normal::new(0.0, 1.0).unwrap();
        let n_d2 = n.cdf(d2);
        let n_d4 = n.cdf(d4);
        let n_prime_d1 = n.pdf(d1);
        let n_prime_d3 = n.pdf(d3);

        let (alpha, beta) = match option_type {
            OptionType::Call => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
            OptionType::Put => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        };

        let kappa = if b >= r || b <= epsilon {
            0.0
        } else {
            2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
        };

        let theta = match option_type {
            OptionType::Call => {
                -s * n_prime_d1 * sigma / (2.0 * t_sqrt) - r * k * E.powf(-r * t) * n_d2 + r * b_t_infinity * (s / b_t_infinity).powf(x) * (n_d2 - kappa)
            },
            OptionType::Put => {
                -s * n_prime_d3 * sigma / (2.0 * t_sqrt) + r * k * E.powf(-r * t) * n_d4 - r * b_t_zero * (s / b_t_zero).powf(-x) * (n_d4 + kappa)
            },
        };

        Ok(theta / 365.0)
    }

    /// Calculate the vega (sensitivity to volatility) of an American option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// The vega of the American option
    pub fn vega(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        q: f64,
        option_type: OptionType,
    ) -> Result<f64, OptionError> {
        let epsilon = 0.00001;
        let t_sqrt = t.sqrt();
        let b = q;
        let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
        let b_infinity = beta / (beta - 1.0) * k;
        let b_zero = match option_type {
            OptionType::Call => k.max(r / (r - b) * k),
            OptionType::Put => k.min(r / (r - b) * k),
        };
        let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
        let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t) + epsilon));
        let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
        let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

        let (d1, _, d3, _) = (
            (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt),
            (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt),
            ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt),
            ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt),
        );

        let n = Normal::new(0.0, 1.0).unwrap();
        let n_prime_d1 = if d1.abs() > 10.0 { 1e-10 } else { n.pdf(d1) };
        let n_prime_d3 = if d3.abs() > 10.0 { 1e-10 } else { n.pdf(d3) };
        let (_, beta) = match option_type {
            OptionType::Call => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
            OptionType::Put => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        };

        let vega = match option_type {
            OptionType::Call => {
                s * t_sqrt * n_prime_d1 * (1.0 - (s / b_t_infinity).powf(beta) * n.cdf(y))
            }
            OptionType::Put => {
                s * t_sqrt * n_prime_d3 * (1.0 - (s / b_t_zero).powf(-beta) * n.cdf(-y))
            }
        };
        Ok(vega)
    }

    /// Calculate the rho (sensitivity to interest rate) of an American option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// The rho of the American option (per 1% change in interest rate)
    pub fn rho(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        q: f64,
        option_type: OptionType,
    ) -> Result<f64, OptionError> {
        let epsilon = 0.00001;
        let t_sqrt = t.sqrt();
        let b = q;
        let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
        let b_infinity = beta / (beta - 1.0) * k;
        let b_zero = match option_type {
            OptionType::Call => k.max(r / (r - b) * k),
            OptionType::Put => k.min(r / (r - b) * k),
        };

        let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
        let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
        let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
        let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
        let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

        let d2 = (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
        let d4 = ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);

        let n = Normal::new(0.0, 1.0).unwrap();
        let n_d2 = n.cdf(d2);
        let n_d4 = n.cdf(d4);

        let (alpha, beta) = match option_type {
            OptionType::Call => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
            OptionType::Put => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        };

        let kappa = if b >= r || b <= epsilon {
            0.0
        } else {
            2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
        };

        let rho = match option_type {
            OptionType::Call => {
                k * t * E.powf(-r * t) * n_d2 + t * b_t_infinity * (s / b_t_infinity).powf(x) * (n_d2 - kappa)
            },
            OptionType::Put => {
                -k * t * E.powf(-r * t) * n_d4 - t * b_t_zero * (s / b_t_zero).powf(-x) * (n_d4 + kappa)
            },
        };

        Ok(rho / 100.0)
    }

    /// Calculate all Greeks for an American option.
    ///
    /// # Arguments
    ///
    /// * `s` - The current price of the underlying asset
    /// * `k` - The strike price of the option
    /// * `t` - The time to expiration in years
    /// * `r` - The risk-free interest rate
    /// * `sigma` - The volatility of the underlying asset
    /// * `q` - The dividend yield of the underlying asset
    /// * `option_type` - The type of option (Call or Put)
    ///
    /// # Returns
    ///
    /// A `Greeks` struct containing delta, gamma, theta, vega, and rho
    pub fn all_greeks(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: OptionType) -> Result<Greeks, OptionError> {
        Ok(Greeks {
            delta: delta(s, k, t, r, sigma, q, option_type)?,
            gamma: gamma(s, k, t, r, sigma, q, option_type)?,
            theta: theta(s, k, t, r, sigma, q, option_type)?,
            vega: vega(s, k, t, r, sigma, q, option_type)?,
            rho: rho(s, k, t, r, sigma, q, option_type)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes_call_delta() {
        let delta = black_scholes::delta(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
        assert!(delta > 0.5 && delta < 0.7);
    }

    #[test]
    fn test_black_scholes_put_delta() {
        let delta = black_scholes::delta(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Put).unwrap();
        assert!(delta < 0.0 && delta > -0.5);
    }

    #[test]
    fn test_black_scholes_gamma() {
        let gamma = black_scholes::gamma(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
        assert!(gamma > 0.0);
    }

    #[test]
    fn test_black_scholes_all_greeks() {
        let greeks = black_scholes::all_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
        assert!(greeks.delta > 0.0);
        assert!(greeks.gamma > 0.0);
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_bjerksund_stensland_call_delta() {
        let delta = bjerksund_stensland::delta(100.0, 100.0, 1.0, 0.05, 0.2, 0.02, OptionType::Call).unwrap();
        assert!(delta > 0.0 && delta < 1.0);
    }
}
