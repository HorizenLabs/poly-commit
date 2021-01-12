use algebra::{PrimeField, ToConstraintField};

/// Implementation of FiatShamirRng using Poseidon Sponge
pub mod poseidon;

/// Implementation of FiatShamirRng using a ChaChaRng
pub mod chacha;

/// the trait for Fiat-Shamir RNG
pub trait FiatShamirRng<F: PrimeField, ConstraintF: PrimeField> {
    /// initialize the RNG
    fn new() -> Self;

    /// take in field elements
    fn absorb_nonnative_field_elements(&mut self, elems: &[F]);
    /// take in field elements
    fn absorb_native_field_elements<T: ToConstraintField<ConstraintF>>(&mut self, elems: &[T]);
    /// take in bytes
    fn absorb_bytes(&mut self, elems: &[u8]);

    /// take out field elements
    fn squeeze_nonnative_field_elements(&mut self, num: usize) -> Vec<F>;
    /// take in field elements
    fn squeeze_native_field_elements(&mut self, num: usize) -> Vec<ConstraintF>;
    /// take out field elements of 128 bits
    fn squeeze_128_bits_nonnative_field_elements(&mut self, num: usize) -> Vec<F>;
}