use algebra::{PrimeField, ToConstraintField, FpParameters};
use primitives::{PoseidonParameters, PoseidonSBox, PoseidonHash, AlgebraicSponge};
use crate::fiat_shamir::FiatShamirRng;

impl<F, ConstraintF, P, SB> FiatShamirRng<F, ConstraintF> for PoseidonHash<ConstraintF, P, SB>
where
    F: PrimeField,
    ConstraintF: PrimeField,
    P: PoseidonParameters<Fr = ConstraintF>,
    SB: PoseidonSBox<P>
{
    fn new() -> Self {
        <Self as AlgebraicSponge<ConstraintF>>::new()
    }

    fn absorb_nonnative_field_elements(&mut self, elems: &[F]) {
        // Serialize elems to bits
        let elems_bits: Vec<bool> = elems
            .iter()
            .flat_map(|fe|{ fe.write_bits() })
            .collect();

        // Pack (safely) elems_bits into native field elements
        let native_fes = elems_bits.as_slice().to_field_elements().unwrap();

        // Absorb native field elements
        self.absorb(native_fes);
    }

    fn absorb_native_field_elements<T: ToConstraintField<ConstraintF>>(&mut self, elems: &[T]) {
        self.absorb(elems.iter().flat_map(|t| t.to_field_elements().unwrap()).collect())
    }

    fn absorb_bytes(&mut self, elems: &[u8]) {
        self.absorb(elems.to_field_elements().unwrap())
    }

    fn squeeze_nonnative_field_elements(&mut self, num: usize) -> Vec<F> {
        let nonnative_bits = F::Params::MODULUS_BITS as usize;
        let required_bits = num * nonnative_bits;
        let required_native_fes = ((required_bits/ConstraintF::Params::MODULUS_BITS as usize) as f64).ceil() as usize;
        let native_squeeze = self.squeeze(required_native_fes);
        let native_bits = native_squeeze
            .into_iter()
            .flat_map(|native_fe| { native_fe.write_bits() })
            .collect::<Vec<_>>();
        native_bits.to_field_elements().unwrap().into_iter().take(num).collect()
    }

    fn squeeze_native_field_elements(&mut self, num: usize) -> Vec<ConstraintF> {
        self.squeeze(num)
    }

    fn squeeze_128_bits_nonnative_field_elements(&mut self, num: usize) -> Vec<F> {

        // Compute number of challenges we can extract from a single field element
        let modulus_bits = F::Params::MODULUS_BITS as usize;
        assert!(modulus_bits >= 128);
        let challenges_per_fe = modulus_bits/128;

        // Compute number of field elements we need in order to provide the requested 'num'
        // 128 bits field elements
        let to_squeeze = ((num/challenges_per_fe) as f64).ceil() as usize;

        // Squeeze the required number of field elements
        let outputs_bits = self
            .squeeze_nonnative_field_elements(to_squeeze)
            .into_iter()
            // Take only up to capacity bits for each field element for safety reasons
            .flat_map(|fe: F|{ fe.write_bits() }).collect::<Vec<_>>();

        // Take the required amount of 128 bits chunks and read (safely) a non-native field
        // element out of each of them.
        outputs_bits.chunks(128).take(num).map(|bits| {
            F::read_bits(bits.to_vec()).expect("Should be able to read a nonnativefield element from 128 bits")
        }).collect()
    }
}