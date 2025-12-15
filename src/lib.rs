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

/// Bjerksund-Stensland 2002 model for American options
///
/// This implementation uses the Bjerksund-Stensland 2002 approximation for pricing
/// American options, with Greeks computed via numerical differentiation of the price.
pub mod bjerksund_stensland {
    use super::*;

    /// Helper function phi() from the Bjerksund-Stensland 2002 paper
    ///
    /// φ(S, T, γ, H, I) = e^λ × S^γ × [N(d) - (I/S)^κ × N(d - 2ln(I/S)/(σ√T))]
    /// where:
    ///   λ = -rT + γbT + 0.5γ(γ-1)σ²T
    ///   d = -[ln(S/H) + (b + (γ - 0.5)σ²)T] / (σ√T)
    ///   κ = 2b/σ² + (2γ - 1)
    fn phi(s: f64, t: f64, gamma: f64, h: f64, i: f64, r: f64, b: f64, sigma: f64) -> f64 {
        let sigma_sq = sigma.powi(2);
        let lambda = (-r * t + gamma * b * t + 0.5 * gamma * (gamma - 1.0) * sigma_sq * t).exp();
        // Note: d uses ln(S/H), not ln(H/S)
        let d = -((s / h).ln() + (b + (gamma - 0.5) * sigma_sq) * t) / (sigma * t.sqrt());
        let kappa = 2.0 * b / sigma_sq + (2.0 * gamma - 1.0);

        let n = Normal::new(0.0, 1.0).unwrap();

        lambda * s.powf(gamma) * (n.cdf(d) - (i / s).powf(kappa) * n.cdf(d - 2.0 * (i / s).ln() / (sigma * t.sqrt())))
    }

    /// European call price using generalized Black-Scholes
    fn european_call(s: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64) -> f64 {
        let sigma_sq = sigma.powi(2);
        let d1 = ((s / k).ln() + (b + 0.5 * sigma_sq) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        let n = Normal::new(0.0, 1.0).unwrap();
        s * ((b - r) * t).exp() * n.cdf(d1) - k * (-r * t).exp() * n.cdf(d2)
    }

    /// Calculate the price of an American call option using Bjerksund-Stensland 2002
    fn call_price(s: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64) -> f64 {
        // If no time to expiry, return intrinsic value
        if t <= 0.0 {
            return (s - k).max(0.0);
        }

        // If b >= r, American call = European call (no early exercise benefit)
        if b >= r {
            return european_call(s, k, t, r, b, sigma);
        }

        let sigma_sq = sigma.powi(2);

        // Calculate beta (must be > 1)
        let beta = (0.5 - b / sigma_sq) + ((b / sigma_sq - 0.5).powi(2) + 2.0 * r / sigma_sq).sqrt();

        // B_infinity = β/(β-1) * K
        let b_infinity = beta / (beta - 1.0) * k;

        // B_0 = max(K, r/(r-b) * K)
        let b_0 = k.max(r / (r - b) * k);

        // h(T) = -(bT + 2σ√T) * B_0 / (B_∞ - B_0)
        let h_t = -(b * t + 2.0 * sigma * t.sqrt()) * b_0 / (b_infinity - b_0);

        // I = B_0 + (B_∞ - B_0) * (1 - e^h(T))
        let i = b_0 + (b_infinity - b_0) * (1.0 - h_t.exp());

        // If S >= I, exercise immediately
        if s >= i {
            return s - k;
        }

        // t1 = 0.5 * (sqrt(5) - 1) * T (golden ratio split)
        let t1 = 0.5 * (5.0_f64.sqrt() - 1.0) * t;

        // h(t1) and I1
        let h_t1 = -(b * t1 + 2.0 * sigma * t1.sqrt()) * b_0 / (b_infinity - b_0);
        let i1 = b_0 + (b_infinity - b_0) * (1.0 - h_t1.exp());

        // Alpha values
        let alpha1 = (i - k) * i.powf(-beta);
        let alpha2 = (i1 - k) * i1.powf(-beta);

        // Bjerksund-Stensland 2002 American call price formula
        let bs_price = alpha2 * phi(s, t1, beta, i1, i1, r, b, sigma)
            - alpha2 * phi(s, t1, beta, i, i1, r, b, sigma)
            + phi(s, t1, 1.0, i, i1, r, b, sigma)
            - phi(s, t1, 1.0, k, i1, r, b, sigma)
            - k * phi(s, t1, 0.0, i, i1, r, b, sigma)
            + k * phi(s, t1, 0.0, k, i1, r, b, sigma)
            + alpha1 * phi(s, t, beta, i, i1, r, b, sigma)
            - alpha1 * phi(s, t, beta, i1, i1, r, b, sigma)
            + phi(s, t, 1.0, i1, i1, r, b, sigma)
            - phi(s, t, 1.0, i, i1, r, b, sigma)
            - k * phi(s, t, 0.0, i1, i1, r, b, sigma)
            + k * phi(s, t, 0.0, i, i1, r, b, sigma);

        // American option price must be at least the European price
        let euro_price = european_call(s, k, t, r, b, sigma);
        bs_price.max(euro_price)
    }

    /// Calculate the price of an American put option using put-call transformation
    fn put_price(s: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64) -> f64 {
        // Use put-call transformation:
        // American put P(S, K, r, b) = American call C(K, S, r-b, -b)
        call_price(k, s, t, r - b, -b, sigma)
    }

    /// Calculate the price of an American option using Bjerksund-Stensland 2002
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
    /// The price of the American option
    pub fn price(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        q: f64,
        option_type: OptionType,
    ) -> Result<f64, OptionError> {
        let b = r - q; // cost of carry

        let price = match option_type {
            OptionType::Call => call_price(s, k, t, r, b, sigma),
            OptionType::Put => put_price(s, k, t, r, b, sigma),
        };

        // Ensure price is at least intrinsic value
        let intrinsic = match option_type {
            OptionType::Call => (s - k).max(0.0),
            OptionType::Put => (k - s).max(0.0),
        };

        Ok(price.max(intrinsic))
    }

    /// Calculate the delta (sensitivity to the underlying price) of an American option
    /// using numerical differentiation.
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
        let ds = s * 0.0001; // Small perturbation
        let p_up = price(s + ds, k, t, r, sigma, q, option_type)?;
        let p_down = price(s - ds, k, t, r, sigma, q, option_type)?;
        Ok((p_up - p_down) / (2.0 * ds))
    }

    /// Calculate the gamma (second-order sensitivity to the underlying price) of an American option
    /// using numerical differentiation.
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
        let ds = s * 0.0001; // Small perturbation
        let p_up = price(s + ds, k, t, r, sigma, q, option_type)?;
        let p_mid = price(s, k, t, r, sigma, q, option_type)?;
        let p_down = price(s - ds, k, t, r, sigma, q, option_type)?;
        Ok((p_up - 2.0 * p_mid + p_down) / (ds * ds))
    }

    /// Calculate the theta (sensitivity to time to expiration) of an American option
    /// using numerical differentiation.
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
        let dt = 1.0 / 365.0; // One day
        let p_now = price(s, k, t, r, sigma, q, option_type)?;
        let p_later = price(s, k, (t - dt).max(0.0001), r, sigma, q, option_type)?;
        // Theta is the change in price as time decreases (negative dt)
        Ok(p_later - p_now)
    }

    /// Calculate the vega (sensitivity to volatility) of an American option
    /// using numerical differentiation.
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
    /// The vega of the American option (per 1% change in volatility)
    pub fn vega(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        q: f64,
        option_type: OptionType,
    ) -> Result<f64, OptionError> {
        let d_sigma = 0.0001; // Small perturbation
        let p_up = price(s, k, t, r, sigma + d_sigma, q, option_type)?;
        let p_down = price(s, k, t, r, sigma - d_sigma, q, option_type)?;
        // Vega: dP/d_sigma (unscaled, same convention as Black-Scholes)
        Ok((p_up - p_down) / (2.0 * d_sigma))
    }

    /// Calculate the rho (sensitivity to interest rate) of an American option
    /// using numerical differentiation.
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
        let dr = 0.0001; // Small perturbation
        let p_up = price(s, k, t, r + dr, sigma, q, option_type)?;
        let p_down = price(s, k, t, r - dr, sigma, q, option_type)?;
        // Rho per 1% change
        Ok((p_up - p_down) / (2.0 * dr) / 100.0)
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
