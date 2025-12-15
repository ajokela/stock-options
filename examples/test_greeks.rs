use stock_options::{black_scholes, bjerksund_stensland, OptionType};

fn main() {
    let s = 100.0;
    let k = 95.0;
    let t = 1.0;
    let r = 0.05;
    let sigma = 0.25;
    let q = 0.02;
    
    println!("Parameters: s={}, k={}, t={}, r={}, sigma={}, q={}\n", s, k, t, r, sigma, q);
    
    // Black-Scholes Put
    let bs_put = black_scholes::all_greeks(s, k, t, r, sigma, q, OptionType::Put).unwrap();
    println!("Black-Scholes Put:");
    println!("  Delta: {:.6}", bs_put.delta);
    println!("  Gamma: {:.6}", bs_put.gamma);
    println!("  Theta: {:.6}", bs_put.theta);
    println!("  Vega:  {:.6}", bs_put.vega);
    println!("  Rho:   {:.6}", bs_put.rho);
    
    // American Put
    let am_put = bjerksund_stensland::all_greeks(s, k, t, r, sigma, q, OptionType::Put).unwrap();
    println!("\nBjerksund-Stensland American Put:");
    println!("  Delta: {:.6}", am_put.delta);
    println!("  Gamma: {:.6}", am_put.gamma);
    println!("  Theta: {:.6}", am_put.theta);
    println!("  Vega:  {:.6}", am_put.vega);
    println!("  Rho:   {:.6}", am_put.rho);
    
    // Sanity check
    println!("\nSanity checks:");
    println!("  American put delta in [-1, 0]: {}", am_put.delta >= -1.0 && am_put.delta <= 0.0);
    println!("  American put gamma > 0: {}", am_put.gamma > 0.0);
}
