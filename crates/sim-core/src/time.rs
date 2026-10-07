//! Global time and exact rationals.

/// Cycles at the reference frequency. The one global clock of all planes.
pub type Cycle = u64;

/// A positive rational `num / den`, kept in lowest terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ratio {
    /// Numerator.
    pub num: u64,
    /// Denominator; never 0.
    pub den: u64,
}

impl Ratio {
    /// The value 1.
    pub const ONE: Ratio = Ratio { num: 1, den: 1 };

    /// Panics if `den` is zero.
    pub fn new(num: u64, den: u64) -> Self {
        assert!(den != 0, "ratio with zero denominator");
        let g = gcd(num as u128, den as u128) as u64;
        Self {
            num: num / g,
            den: den / g,
        }
    }

    /// The integer `n` as a ratio `n / 1`.
    pub fn int(n: u64) -> Self {
        Self::new(n, 1)
    }

    /// The value as a float (for cost models and reports).
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// The smaller of `self` and `other`.
    pub fn min(self, other: Ratio) -> Ratio {
        let (a, b) = (
            self.num as u128 * other.den as u128,
            other.num as u128 * self.den as u128,
        );
        if a <= b {
            self
        } else {
            other
        }
    }
}

impl Default for Ratio {
    fn default() -> Self {
        Self::ONE
    }
}

pub(crate) fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
