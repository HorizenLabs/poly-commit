use algebra::{PrimeField, ToConstraintField, FpParameters, BigInteger};
use primitives::AlgebraicSponge;
use r1cs_std::{
    fields::nonnative::{
        nonnative_field_gadget::NonNativeFieldGadget,
        params::get_params
    },
    overhead,
};

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

    /// take in non-native field elements
    fn absorb_nonnative_field_elements(&mut self, elems: &[F]);
    /// take in field elements
    fn absorb_native_field_elements<T: ToConstraintField<ConstraintF>>(&mut self, elems: &[T]);
    /// take in bytes
    fn absorb_bytes(&mut self, elems: &[u8]);

    /// take out non-native field elements
    fn squeeze_nonnative_field_elements(&mut self, num: usize) -> Vec<F>;
    /// take out field elements
    fn squeeze_native_field_elements(&mut self, num: usize) -> Vec<ConstraintF>;
    /// take out non-native field elements of 128 bits
    fn squeeze_128_bits_nonnative_field_elements(&mut self, num: usize) -> Vec<F>;
}

/// compress every two elements if possible. Provides a vector of (limb, num_of_additions),
/// both of which are ConstraintF.
pub(crate) fn compress_elements<
    SimulationF: PrimeField,
    ConstraintF: PrimeField
>(src_limbs: &[(ConstraintF, ConstraintF)]) -> Vec<ConstraintF>
{
    let capacity = ConstraintF::size_in_bits() - 1;
    let mut dest_limbs = Vec::<ConstraintF>::new();

    let params = get_params(SimulationF::size_in_bits(), ConstraintF::size_in_bits());

    let adjustment_factor_lookup_table = {
        let mut table = Vec::<ConstraintF>::new();

        let mut cur = ConstraintF::one();
        for _ in 1..=capacity {
            table.push(cur);
            cur.double_in_place();
        }

        table
    };

    let mut i = 0;
    let src_len = src_limbs.len();
    while i < src_len {
        let first = &src_limbs[i];
        let second = if i + 1 < src_len {
            Some(&src_limbs[i + 1])
        } else {
            None
        };

        let first_max_bits_per_limb = params.bits_per_limb + overhead!(first.1 + &ConstraintF::one());
        let second_max_bits_per_limb = if let Some(second) = second {
            params.bits_per_limb + overhead!(second.1 + &ConstraintF::one())
        } else {
            0
        };

        if let Some(second) = second {
            if first_max_bits_per_limb + second_max_bits_per_limb <= capacity {
                let adjustment_factor =
                    &adjustment_factor_lookup_table[second_max_bits_per_limb];

                dest_limbs.push(first.0 * adjustment_factor + &second.0);
                i += 2;
            } else {
                dest_limbs.push(first.0);
                i += 1;
            }
        } else {
            dest_limbs.push(first.0);
            i += 1;
        }
    }

    dest_limbs
}

/// Push elements to sponge, treated in the non-native field representations.
pub(crate) fn push_elements_to_sponge<
    SimulationF: PrimeField,
    ConstraintF: PrimeField,
    S: AlgebraicSponge<ConstraintF>
>(sponge: &mut S, src: &[SimulationF]) {
    let mut src_limbs = Vec::<(ConstraintF, ConstraintF)>::new();

    for elem in src.iter() {
        let limbs =
            NonNativeFieldGadget::<SimulationF, ConstraintF>::get_limbs_representations(elem).unwrap();
        for limb in limbs.iter() {
            src_limbs.push((*limb, ConstraintF::one()));
            // specifically set to one, since most gadgets in the constraint world would
            // not have zero noise (due to the relatively weak normal form testing in `alloc`)
        }
    }

    let dest_limbs = compress_elements::<SimulationF, ConstraintF>(&src_limbs);
    sponge.absorb(dest_limbs);
}

/// Squeeze num_bits from the Sponge.
pub(crate) fn get_bits_from_sponge<
    ConstraintF: PrimeField,
    S: AlgebraicSponge<ConstraintF>
>(sponge: &mut S, num_bits: usize) -> Vec<bool>
{
    let bits_per_element = ConstraintF::size_in_bits() - 1;
    let num_elements = (num_bits + bits_per_element - 1) / bits_per_element;

    let src_elements = sponge.squeeze(num_elements);
    let mut dest_bits = Vec::<bool>::new();

    let skip = (ConstraintF::Params::REPR_SHAVE_BITS + 1) as usize;
    for elem in src_elements.iter() {
        // discard the highest bit
        let elem_bits = elem.into_repr().to_bits();
        dest_bits.extend_from_slice(&elem_bits[skip..]);
    }

    dest_bits
}

/// Squeeze SimulationF elements from the Sponge.
pub(crate) fn get_elements_from_sponge<
    SimulationF: PrimeField,
    ConstraintF: PrimeField,
    S: AlgebraicSponge<ConstraintF>
>(
    sponge: &mut S,
    num_elements: usize,
    outputs_short_elements: bool,
) -> Vec<SimulationF> {
    let num_bits_per_nonnative = if outputs_short_elements {
        128
    } else {
        SimulationF::size_in_bits() - 1 // also omit the highest bit
    };
    let bits = get_bits_from_sponge::<ConstraintF, S>(sponge, num_bits_per_nonnative * num_elements);

    let mut dest_elements = Vec::<SimulationF>::new();
    bits.chunks_exact(num_bits_per_nonnative).take(num_elements)
        .for_each(|per_nonnative_bits| {
            dest_elements.push(SimulationF::read_bits(per_nonnative_bits.to_vec()).unwrap());
        });

    dest_elements
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