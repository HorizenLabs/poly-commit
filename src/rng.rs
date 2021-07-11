use crate::Vec;
use algebra::{FromBytes, ToBytes, Field, UniformRand};
use std::marker::PhantomData;
use digest::{generic_array::GenericArray, Digest};
use rand_chacha::ChaChaRng;
use rand_core::{RngCore, SeedableRng};

/// General trait for `SeedableRng` that refreshes its seed by hashing together the previous seed
/// and the new seed material.
// TODO: later: re-evaluate decision about ChaChaRng
pub trait FiatShamirRng: RngCore {
    /// Internal State
    type State: Clone;

    /// Create a new `Self` by initializing its internal state with a fresh `seed`,
    /// generically being something serializable to a byte array.
    fn from_seed<'a, T: 'a + ToBytes>(seed: &'a T) -> Self;

    /// Refresh the internal state with new material `seed`, generically being
    /// something serializable to a byte array.
    fn absorb<'a, T: 'a + ToBytes>(&mut self, seed: &'a T);

    /// Squeeze a new random field element
    fn squeeze_128_bits_challenge<F: Field>(&mut self) -> F {
        u128::rand(self).into()
    }

    /// Get the internal state in the form of an instance of `Self::Seed`.
    fn get_state(&self) -> &Self::State;

    /// Set interal state according to the specified `new_seed`
    fn set_state(&mut self, new_state: Self::State);
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

    type State = GenericArray<u8, D::OutputSize>;

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
    #[inline]
    fn from_seed<'a, T: 'a + ToBytes>(seed: &'a T) -> Self {
        let mut bytes = Vec::new();
        seed.write(&mut bytes).unwrap_or(());
        let seed = D::digest(&bytes);
        let r_seed: [u8; 32] = FromBytes::read(seed.as_ref()).unwrap_or([0u8; 32]);
        let r = ChaChaRng::from_seed(r_seed);
        Self {
            r,
            seed,
            digest: PhantomData,
        }
    }

    /// Get `self.seed`.
    #[inline]
    fn get_state(&self) -> &Self::State {
        &self.seed
    }

    /// Set `self.seed` to the specified value
    #[inline]
    fn set_state(&mut self, new_state: Self::State) {
        self.seed = new_state
    }
}