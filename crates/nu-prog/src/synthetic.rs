//! A synthetic AdaMoE router trace with the measured statistics (workload
//! 2): per layer a Dirichlet(alpha) share over the routed experts (alpha 4
//! gives a mean max share of 0.41 of 51 tokens and all 4 experts live in
//! 99% of (step, layer); measured 0.41 and 97%), tokens drawn from it at
//! step 0, and each later step keeping a token's expert with probability
//! `step_agree` (measured 0.91). Layers are independent (measured layer to
//! layer agreement is at chance).

use std::collections::BTreeMap;

use crate::adamoe::Params;
use crate::label;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// Gamma(a, 1), a >= 1 (Marsaglia and Tsang).
    fn gamma(&mut self, a: f64) -> f64 {
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 && self.unit().ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    fn pick(&mut self, shares: &[f64]) -> u8 {
        let u = self.unit();
        let mut acc = 0.0;
        for (i, s) in shares.iter().enumerate() {
            acc += s;
            if u < acc {
                return i as u8;
            }
        }
        (shares.len() - 1) as u8
    }
}

/// Router records keyed by `label::router(step, layer)`: one expert id
/// (u8) per token.
pub fn synthetic_trace(
    p: &Params,
    seed: u64,
    alpha: f64,
    step_agree: f64,
) -> BTreeMap<u64, Vec<u8>> {
    assert!(alpha >= 1.0, "alpha must be at least 1");
    let mut rng = Rng(seed);
    let mut out = BTreeMap::new();
    let e = p.experts as usize;
    for l in 0..p.layers {
        let g: Vec<f64> = (0..e).map(|_| rng.gamma(alpha)).collect();
        let sum: f64 = g.iter().sum();
        let shares: Vec<f64> = g.iter().map(|x| x / sum).collect();
        let mut prev: Option<Vec<u8>> = None;
        for s in 0..p.steps {
            let ids: Vec<u8> = (0..p.tokens as usize)
                .map(|t| match &prev {
                    Some(pv) if rng.unit() < step_agree => pv[t],
                    _ => rng.pick(&shares),
                })
                .collect();
            out.insert(label::router(s, l), ids.clone());
            prev = Some(ids);
        }
    }
    out
}
