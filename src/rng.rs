use crate::Vec;
use algebra::{FromBytes, ToBytes, to_bytes, Field, UniformRand};
use std::marker::PhantomData;
use digest::{generic_array::GenericArray, Digest};
use rand_chacha::ChaChaRng;
use rand_core::{RngCore, SeedableRng};

/// General trait for `SeedableRng` that refreshes its seed by hashing together the previous seed
/// and the new seed material.
// TODO: later: re-evaluate decision about ChaChaRng
pub trait FiatShamirRng: RngCore {
    /// initialize the RNG
    fn new() -> Self;

    /// Create a new `Self` by initializing with a fresh seed.
    fn from_seed<'a, T: 'a + ToBytes>(seed: &'a T) -> Self;

    /// Get `self.seed`.
    fn get_seed(&self) -> &[u8];

    /// Refresh `self.seed` with new material. Achieved by setting
    /// `self.seed = H(self.seed || new_seed)`.
    fn absorb<'a, T: 'a + ToBytes>(&mut self, seed: &'a T);

    /// Squeeze a new random field element
    fn squeeze_128_bits_challenge<F: Field>(&mut self) -> F {
        u128::rand(self).into()
    }
}

/// A `SeedableRng` that refreshes its seed by hashing together the previous seed
/// and the new seed material.
// TODO: later: re-evaluate decision about ChaChaRng
pub struct FiatShamirChaChaRng<D: Digest> {
    r: ChaChaRng,
    seed: GenericArray<u8, D::OutputSize>,
    #[doc(hidden)]
    digest: PhantomData<D>,
}

impl<D: Digest> RngCore for FiatShamirChaChaRng<D> {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.r.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.r.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.r.fill_bytes(dest);
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        Ok(self.r.fill_bytes(dest))
    }
}

impl<D: Digest> FiatShamirRng for FiatShamirChaChaRng<D> {
    fn new() -> Self {
        let seed = [0u8; 32];
        Self::from_seed(&to_bytes![seed].unwrap())
    }

    /// Refresh `self.seed` with new material. Achieved by setting
    /// `self.seed = H(self.seed || new_seed)`.
    #[inline]
    fn absorb<'a, T: 'a + ToBytes>(&mut self, seed: &'a T) {
        let mut bytes = Vec::new();
        seed.write(&mut bytes).expect("failed to convert to bytes");
        bytes.extend_from_slice(&self.seed);
        self.seed = D::digest(&bytes);
        let seed: [u8; 32] = FromBytes::read(self.seed.as_ref()).expect("failed to get [u32; 8]");
        self.r = ChaChaRng::from_seed(seed);
    }

    /// Create a new `Self` by initializing with a fresh seed.
    /// `self.seed = H(self.seed || new_seed)`.
    #[inline]
    fn from_seed<'a, T: 'a + ToBytes>(seed: &'a T) -> Self {
        let mut bytes = Vec::new();
        seed.write(&mut bytes).expect("failed to convert to bytes");
        let seed = D::digest(&bytes);
        let r_seed: [u8; 32] = FromBytes::read(seed.as_ref()).expect("failed to get [u32; 8]");
        let r = ChaChaRng::from_seed(r_seed);
        Self {
            r,
            seed,
            digest: PhantomData,
        }
    }

    /// Get `self.seed`.
    #[inline]
    fn get_seed(&self) -> &[u8] {
        &self.seed
    }
}