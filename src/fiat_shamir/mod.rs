use algebra::{PrimeField, ToConstraintField};

/// Implementation of FiatShamirRng using Poseidon Sponge
pub mod poseidon;

/// Implementation of FiatShamirRng using a ChaChaRng
pub mod chacha;

/// Gadgets to enforce FiatShamirRng on R1CS circuits
pub mod constraints;

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


#[cfg(test)]
mod test {
    use algebra::{PrimeField, FpParameters, leading_zeros};
    use crate::fiat_shamir::FiatShamirRng;
    use rand::thread_rng;

    pub(crate) fn test_absorb_squeeze_vals<F: PrimeField, ConstraintF: PrimeField, FS: FiatShamirRng<F, ConstraintF>>(
        non_native_inputs: Vec<F>,
        native_inputs: Vec<ConstraintF>,
        byte_inputs: &[u8],
        outputs_for_non_native_inputs: (F, ConstraintF, F),
        outputs_for_native_inputs: (F, ConstraintF, F),
        outputs_for_byte_inputs: (F, ConstraintF, F),
    )
    {
        // Non native inputs
        let mut fs_rng = FS::new();
        fs_rng.absorb_nonnative_field_elements(non_native_inputs.as_slice());
        /*println!(
            "{:?}\n{:?}\n{:?}\n",
            fs_rng.squeeze_nonnative_field_elements(1)[0],
            fs_rng.squeeze_native_field_elements(1)[0],
            fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0]
        );*/
        assert_eq!(
            outputs_for_non_native_inputs,
            (
                fs_rng.squeeze_nonnative_field_elements(1)[0],
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0],
            )
        );

        // Native inputs
        let mut fs_rng = FS::new();
        fs_rng.absorb_native_field_elements(native_inputs.as_slice());
        /*println!(
            "{:?}\n{:?}\n{:?}\n",
            fs_rng.squeeze_nonnative_field_elements(1)[0],
            fs_rng.squeeze_native_field_elements(1)[0],
            fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0]
        );*/
        assert_eq!(
            outputs_for_native_inputs,
            (
                fs_rng.squeeze_nonnative_field_elements(1)[0],
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0],
            )
        );

        // Byte inputs
        let mut fs_rng = FS::new();
        fs_rng.absorb_bytes(byte_inputs);
        /*println!(
            "{:?}\n{:?}\n{:?}\n",
            fs_rng.squeeze_nonnative_field_elements(1)[0],
            fs_rng.squeeze_native_field_elements(1)[0],
            fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0]
        );*/
        assert_eq!(
            outputs_for_byte_inputs,
            (
                fs_rng.squeeze_nonnative_field_elements(1)[0],
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0],
            )
        );
    }

    pub(crate) fn test_squeeze_consistency<F: PrimeField, ConstraintF: PrimeField, FS: FiatShamirRng<F, ConstraintF>>(){
        let rng = &mut thread_rng();

        let input_size = 5;
        let to_squeeze_max = 5;

        let input = vec![ConstraintF::rand(rng); input_size];
        let mut fs_rng = FS::new();
        fs_rng.absorb_native_field_elements(&input);

        for i in 1..=to_squeeze_max {
            let native_outputs = fs_rng.squeeze_native_field_elements(i);
            let non_native_outputs = fs_rng.squeeze_nonnative_field_elements(i);
            let non_native_128_bits_outputs = fs_rng.squeeze_128_bits_nonnative_field_elements(i);

            // Test squeezing of correct number of field elements
            assert_eq!(i, native_outputs.len());
            assert_eq!(i, non_native_outputs.len());
            assert_eq!(i, non_native_128_bits_outputs.len());

            // Test validity of each of the outputs
            native_outputs.into_iter().for_each(|out| { assert!(out.is_valid()); });
            non_native_outputs.into_iter().for_each(|out| { assert!(out.is_valid()); });
            non_native_128_bits_outputs.into_iter().for_each(|out| {
                assert!(out.is_valid());
                let expected_zeros = F::Params::MODULUS_BITS as usize - 128usize;
                assert!(leading_zeros(out.write_bits().as_slice()) as usize >= expected_zeros);
            });
        }
    }
}