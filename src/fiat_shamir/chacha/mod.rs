use algebra::{PrimeField, ToConstraintField};
use digest::Digest;
use rand_core::{
    RngCore, SeedableRng
};
use rand_chacha::ChaChaRng;
use crate::fiat_shamir::FiatShamirRng;
use std::marker::PhantomData;

/// use a ChaCha stream cipher to generate the actual pseudorandom bits
/// use a digest funcion to do absorbing
pub struct FiatShamirChaChaRng<F: PrimeField, CF: PrimeField, D: Digest> {
    /// Underlying rng
    pub r: ChaChaRng,
    /// Seed for rng
    pub seed: Vec<u8>,
    #[doc(hidden)]
    field: PhantomData<F>,
    constraint_field: PhantomData<CF>,
    digest: PhantomData<D>,
}

impl<F: PrimeField, CF: PrimeField, D: Digest> FiatShamirRng<F, CF>
for FiatShamirChaChaRng<F, CF, D>
{
    fn new() -> Self {
        let seed = [0; 32];
        let r = ChaChaRng::from_seed(seed);
        Self {
            r,
            seed: seed.to_vec(),
            field: PhantomData,
            constraint_field: PhantomData,
            digest: PhantomData,
        }
    }

    fn absorb_nonnative_field_elements(&mut self, elems: &[F]) {
        let mut bytes = Vec::new();
        for elem in elems {
            elem.write(&mut bytes).expect("failed to convert to bytes");
        }
        self.absorb_bytes(&bytes);
    }

    fn absorb_native_field_elements<T: ToConstraintField<CF>>(&mut self, src: &[T]) {
        let mut elems = Vec::<CF>::new();
        for elem in src.iter() {
            elems.append(&mut elem.to_field_elements().unwrap());
        }

        let mut bytes = Vec::new();
        for elem in elems.iter() {
            elem.write(&mut bytes).expect("failed to convert to bytes");
        }
        self.absorb_bytes(&bytes);
    }

    fn absorb_bytes(&mut self, elems: &[u8]) {
        let mut bytes = elems.to_vec();
        bytes.extend_from_slice(&self.seed);

        let new_seed = D::digest(&bytes);
        self.seed = (*new_seed.as_slice()).to_vec();

        let mut seed = [0u8; 32];
        for (i, byte) in self.seed.as_slice().iter().enumerate() {
            seed[i] = *byte;
        }

        self.r = ChaChaRng::from_seed(seed);
    }

    fn squeeze_nonnative_field_elements(&mut self, num: usize) -> Vec<F> {
        let mut res = Vec::<F>::new();
        for _ in 0..num {
            res.push(F::rand(&mut self.r));
        }
        res
    }

    fn squeeze_native_field_elements(&mut self, num: usize) -> Vec<CF> {
        let mut res = Vec::<CF>::new();
        for _ in 0..num {
            res.push(CF::rand(&mut self.r));
        }
        res
    }

    fn squeeze_128_bits_nonnative_field_elements(&mut self, num: usize) -> Vec<F> {
        let mut res = Vec::<F>::new();
        for _ in 0..num {
            let mut x = [0u8; 16];
            self.r.fill_bytes(&mut x);
            res.push(F::from_random_bytes(&x).unwrap());
        }
        res
    }
}