use algebra::PrimeField;
use primitives::AlgebraicSponge;
use r1cs_crypto::AlgebraicSpongeGadget;
use r1cs_core::{ConstraintSystem, SynthesisError, LinearCombination};
use r1cs_std::{
    alloc::AllocGadget,
    fields::{
        FieldGadget,
        fp::FpGadget, nonnative::{
            params::get_params,
            nonnative_field_gadget::NonNativeFieldGadget
        }
    },
    to_field_gadget_vec::ToConstraintFieldGadget,
    overhead,
    bits::{uint8::UInt8, boolean::Boolean, ToBitsGadget},
};
use std::marker::PhantomData;

/// the trait for Fiat-Shamir RNG Gadget
pub trait FiatShamirRngGadget<F: PrimeField, ConstraintF: PrimeField>: Sized + Clone {

    /// initialize the RNG
    fn new<CS: ConstraintSystem<ConstraintF>>(cs: CS) -> Result<Self, SynthesisError>;

    /// take in non-native field elements
    fn enforce_absorb_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        elems: &[NonNativeFieldGadget<F, ConstraintF>]
    ) -> Result<(), SynthesisError>;

    /// take in field elements
    fn enforce_absorb_native_field_elements<
        CS: ConstraintSystem<ConstraintF>,
        T: ToConstraintFieldGadget<ConstraintF, FieldGadget = FpGadget<ConstraintF>>>(
            &mut self,
            cs: CS,
            elems: &[T]
    ) -> Result<(), SynthesisError>;

    /// take in bytes
    fn enforce_absorb_bytes<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        elems: &[UInt8]
    ) -> Result<(), SynthesisError>;

    /// Output non-native field elements
    fn enforce_squeeze_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<Vec<NonNativeFieldGadget<F, ConstraintF>>, SynthesisError> {
        Ok(self.enforce_squeeze_nonnative_field_elements_and_bits(cs, num)?.0)
    }

    /// Output non-native field elements and the corresponding bits (this can reduce repeated computation).
    fn enforce_squeeze_nonnative_field_elements_and_bits<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<(Vec<NonNativeFieldGadget<F, ConstraintF>>, Vec<Vec<Boolean>>), SynthesisError>;

    /// Output field elements
    fn enforce_squeeze_native_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<Vec<FpGadget<ConstraintF>>, SynthesisError>;

    /// Output non-native field elements of 128 bits
    fn enforce_squeeze_128_bits_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<Vec<NonNativeFieldGadget<F, ConstraintF>>, SynthesisError> {
        Ok(self.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(cs, num)?.0)
    }

    /// Output non-native field elements with only 128 bits, and the corresponding bits (this can reduce
    /// repeated computation).
    fn enforce_squeeze_128_bits_nonnative_field_elements_and_bits<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize,
    ) -> Result<(Vec<NonNativeFieldGadget<F, ConstraintF>>, Vec<Vec<Boolean>>), SynthesisError>;
}

/// Enforce compression of every two elements if possible.
pub(crate) fn compress_gadgets<
    SimulationF: PrimeField,
    ConstraintF: PrimeField,
    CS: ConstraintSystem<ConstraintF>
>(
    mut cs: CS,
    src_limbs: &[(FpGadget<ConstraintF>, ConstraintF)]
) -> Result<Vec<FpGadget<ConstraintF>>, SynthesisError> 
{
    let capacity = ConstraintF::size_in_bits() - 1;
    let mut dest_limbs = Vec::<FpGadget<ConstraintF>>::new();

    if src_limbs.is_empty() {
        return Ok(vec![]);
    }

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

    let mut i: usize = 0;
    let src_len = src_limbs.len();
    while i < src_len {
        let first = &src_limbs[i];
        let second = if i + 1 < src_len {
            Some(&src_limbs[i + 1])
        } else {
            None
        };

        let first_max_bits_per_limb = params.bits_per_limb + overhead!(first.1 + &ConstraintF::one());
        let second_max_bits_per_limb = if second.is_some() {
            params.bits_per_limb + overhead!(second.unwrap().1 + &ConstraintF::one())
        } else {
            0
        };

        if second.is_some() && first_max_bits_per_limb + second_max_bits_per_limb <= capacity {
            let adjustment_factor = &adjustment_factor_lookup_table[second_max_bits_per_limb];

            dest_limbs.push(
                first.0
                    .mul_by_constant(cs.ns(|| format!("first[{}] * adjustment factor", i)), adjustment_factor)?
                    .add(cs.ns(|| format!("first[{}] * adjustment factor + second[{}]", i, i + 1)), &second.unwrap().0)?
            );
            i += 2;
        } else {
            dest_limbs.push(first.0.clone());
            i += 1;
        }
    }

    Ok(dest_limbs)
}

/// Enforce absorbing non native field element gadgets into the sponge.
pub(crate) fn push_gadgets_to_sponge<
    SimulationF: PrimeField,
    ConstraintF: PrimeField,
    S:  AlgebraicSponge<ConstraintF>,
    SG: AlgebraicSpongeGadget<S, ConstraintF, DataGadget = FpGadget<ConstraintF>>,
    CS: ConstraintSystem<ConstraintF>
>(
    mut cs: CS,
    sponge: &mut SG,
    src: &[NonNativeFieldGadget<SimulationF, ConstraintF>],
) -> Result<(), SynthesisError> {
    let mut src_limbs: Vec<(FpGadget<ConstraintF>, ConstraintF)> = Vec::new();

    for elem in src.iter() {
        for limb in elem.limbs.iter() {
            let num_of_additions_over_normal_form =
                if elem.num_of_additions_over_normal_form == ConstraintF::zero() {
                    ConstraintF::one()
                } else {
                    elem.num_of_additions_over_normal_form
                };
            src_limbs.push((limb.clone(), num_of_additions_over_normal_form));
        }
    }

    let dest_limbs = compress_gadgets::<SimulationF, ConstraintF, _>(
        cs.ns(|| "compress limbs"),
        &src_limbs
    )?;
    sponge.enforce_absorb(cs.ns(|| "absorb compressed limbs"), dest_limbs.as_slice())?;
    Ok(())
}

/// Squeeze num_bits Booleans from the sponge
pub fn get_booleans_from_sponge<
    ConstraintF: PrimeField,
    S:  AlgebraicSponge<ConstraintF>,
    SG: AlgebraicSpongeGadget<S, ConstraintF>,
    CS: ConstraintSystem<ConstraintF>
>(
    mut cs: CS,
    sponge: &mut SG,
    num_bits: usize,
) -> Result<Vec<Boolean>, SynthesisError> {
    let bits_per_element = ConstraintF::size_in_bits() - 1;

    // This is a way to compute ceil(num_bits/bits_per_element)
    let num_elements = (num_bits + bits_per_element - 1) / bits_per_element;

    let src_elements = sponge.enforce_squeeze(
        cs.ns(|| "squeeze required bits"),
        num_elements
    )?;
    let mut dest_bits = Vec::<Boolean>::new();

    for (i, elem) in src_elements.iter().enumerate() {
        // If only few elements are squeezed, one can omit the range proof
        // if needed, as the security loss is just as many bits as elements
        // squeezed.
        let elem_bits = elem.to_bits_strict(
            cs.ns(|| format!("elem {} to bits", i))
        )?;
        dest_bits.extend_from_slice(&elem_bits[1..]); // discard the highest bit
    }

    Ok(dest_bits)
}

/// Squeeze non native field gadgets, and the corresponding Booleans, from the sponge.
pub(crate) fn get_gadgets_and_bits_from_sponge<
    SimulationF: PrimeField,
    ConstraintF: PrimeField,
    S:  AlgebraicSponge<ConstraintF>,
    SG: AlgebraicSpongeGadget<S, ConstraintF>,
    CS: ConstraintSystem<ConstraintF>
>(
    mut cs: CS,
    sponge: &mut SG,
    num_elements: usize,
    outputs_short_elements: bool,
) -> Result<(Vec<NonNativeFieldGadget<SimulationF, ConstraintF>>, Vec<Vec<Boolean>>), SynthesisError> {

    let params = get_params(SimulationF::size_in_bits(), ConstraintF::size_in_bits());

    let num_bits_per_nonnative = if outputs_short_elements {
        128
    } else {
        SimulationF::size_in_bits() - 1 // also omit the highest bit
    };
    let bits = get_booleans_from_sponge(
        cs.ns(|| "squeeze bits from sponge"),
        sponge,
        num_bits_per_nonnative * num_elements
    )?;

    let mut lookup_table = Vec::<Vec<ConstraintF>>::new();
    let mut cur = SimulationF::one();
    for _ in 0..num_bits_per_nonnative {
        let repr = NonNativeFieldGadget::<SimulationF, ConstraintF>::get_limbs_representations(&cur)?;
        lookup_table.push(repr);
        cur.double_in_place();
    }

    let mut dest_gadgets = Vec::<NonNativeFieldGadget<SimulationF, ConstraintF>>::new();
    let mut dest_bits = Vec::<Vec<Boolean>>::new();

    // Pack the Booleans into FpGadget limbs and the limbs into NonNativeFieldGadgets
    bits.chunks_exact(num_bits_per_nonnative).take(num_elements).enumerate()
        .for_each(|(i, per_nonnative_bits)| {
            let mut val = vec![ConstraintF::zero(); params.num_limbs];
            let mut lc = vec![LinearCombination::<ConstraintF>::zero(); params.num_limbs];

            let mut per_nonnative_bits_le = per_nonnative_bits.to_vec();
            per_nonnative_bits_le.reverse();

            dest_bits.push(per_nonnative_bits_le.clone());

            for (j, bit) in per_nonnative_bits_le.iter().enumerate() {
                if bit.get_value().unwrap_or_default() {
                    for (k, val) in val.iter_mut().enumerate().take(params.num_limbs) {
                        *val += &lookup_table[j][k];
                    }
                }

                for k in 0..params.num_limbs {
                    lc[k] = &lc[k] + bit.lc(CS::one(), lookup_table[j][k]);
                }
            }

            let mut limbs = Vec::new();
            for k in 0..params.num_limbs {
                let gadget =
                    FpGadget::alloc(
                        cs.ns(|| format!("alloc {} limb {}", i, k)),
                        || Ok(val[k])
                    ).unwrap();
                lc[k] = gadget.get_variable() - lc[k].clone();
                cs.enforce(
                    || format!("unpacking constraint {} for limb {}", i, k),
                    |lc| lc,
                    |lc| lc,
                    |_| lc[k].clone()
                );
                limbs.push(gadget);
            }
            dest_gadgets.push(
                NonNativeFieldGadget::<SimulationF, ConstraintF> {
                    limbs,
                    num_of_additions_over_normal_form: ConstraintF::zero(),
                    is_in_the_normal_form: true,
                    simulation_phantom: PhantomData,
                }
            );
        });

    Ok((dest_gadgets, dest_bits))
}

#[cfg(test)]
pub(crate) mod test {
    use algebra::PrimeField;
    use crate::fiat_shamir::FiatShamirRng;
    use crate::fiat_shamir::constraints::FiatShamirRngGadget;
    use r1cs_std::test_constraint_system::TestConstraintSystem;
    use r1cs_core::ConstraintSystem;
    use r1cs_std::{
        alloc::AllocGadget,
        fields::{
            fp::FpGadget, FieldGadget,
            nonnative::nonnative_field_gadget::NonNativeFieldGadget
        },
        bits::uint8::UInt8,
    };

    pub(crate) fn test_native_result<
        F: PrimeField,
        ConstraintF: PrimeField,
        FS:  FiatShamirRng<F, ConstraintF>,
        FSG: FiatShamirRngGadget<F, ConstraintF>,
    >(
        non_native_inputs: Vec<F>,
        native_inputs: Vec<ConstraintF>,
        byte_inputs: &[u8],
    )
    {

        let mut cs = TestConstraintSystem::<ConstraintF>::new();

        // Non Native inputs
        let mut fs_rng = FS::new();
        fs_rng.absorb_nonnative_field_elements(non_native_inputs.as_slice());

        // Alloc non native inputs
        let non_native_gs = Vec::<NonNativeFieldGadget<F, ConstraintF>>::alloc(
            cs.ns(|| "alloc non native field elements"),
            || Ok(non_native_inputs)
        ).unwrap();

        let mut fs_rng_g = FSG::new(cs.ns(|| "new fs_rng_g for non native inputs")).unwrap();
        fs_rng_g.enforce_absorb_nonnative_field_elements(
            cs.ns(|| "enforce absorb non native field elements"), non_native_gs.as_slice()
        ).unwrap();
        assert_eq!(
            (
                fs_rng.squeeze_nonnative_field_elements(1)[0],
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0],
            ),
            (
                fs_rng_g.enforce_squeeze_nonnative_field_elements(
                    cs.ns(||"squeeze non native given non native absorb"), 1
                ).unwrap()[0].get_value().unwrap(),
                fs_rng_g.enforce_squeeze_native_field_elements(
                    cs.ns(|| "squeeze native given non native absorb"), 1
                ).unwrap()[0].value.unwrap(),
                fs_rng_g.enforce_squeeze_128_bits_nonnative_field_elements(
                    cs.ns(|| "squeeze 128 bits given non native absorb"), 1
                ).unwrap()[0].get_value().unwrap(),
            )
        );

        // Native inputs
        let mut fs_rng = FS::new();
        fs_rng.absorb_native_field_elements(native_inputs.as_slice());

        // Alloc native inputs
        let native_inputs_gs = Vec::<FpGadget<ConstraintF>>::alloc(
            cs.ns(|| "alloc native inputs"),
            || Ok(native_inputs)
        ).unwrap();

        let mut fs_rng_g = FSG::new(cs.ns(|| "new fs_rng_g for native inputs")).unwrap();
        fs_rng_g.enforce_absorb_native_field_elements(
            cs.ns(|| "enforce absorb native field elements"), &native_inputs_gs
        ).unwrap();
        assert_eq!(
            (
                fs_rng.squeeze_nonnative_field_elements(1)[0],
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0],
            ),
            (
                fs_rng_g.enforce_squeeze_nonnative_field_elements(
                    cs.ns(||"squeeze non native given native absorb"), 1
                ).unwrap()[0].get_value().unwrap(),
                fs_rng_g.enforce_squeeze_native_field_elements(
                    cs.ns(|| "squeeze native given native absorb"), 1
                ).unwrap()[0].value.unwrap(),
                fs_rng_g.enforce_squeeze_128_bits_nonnative_field_elements(
                        cs.ns(|| "squeeze 128 bits given native absorb"), 1
                ).unwrap()[0].get_value().unwrap(),
            )
        );

        // Byte inputs
        let mut fs_rng = FS::new();
        fs_rng.absorb_bytes(byte_inputs);

        // Alloc byte inputs
        let byte_inputs_gs = Vec::<UInt8>::alloc(
            cs.ns(|| "alloc byte inputs"),
            || Ok(byte_inputs.to_vec())
        ).unwrap();

        let mut fs_rng_g = FSG::new(cs.ns(|| "new fs_rng_g for byte inputs")).unwrap();
        fs_rng_g.enforce_absorb_bytes(
            cs.ns(|| "enforce absorb byes"), &byte_inputs_gs[..]
        ).unwrap();

        assert_eq!(
            (
                fs_rng.squeeze_nonnative_field_elements(1)[0],
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0],
            ),
            (
                fs_rng_g.enforce_squeeze_nonnative_field_elements(
                    cs.ns(||"squeeze non native given bytes absorb"), 1
                ).unwrap()[0].get_value().unwrap(),
                fs_rng_g.enforce_squeeze_native_field_elements(
                    cs.ns(|| "squeeze native given bytes absorb"), 1
                ).unwrap()[0].value.unwrap(),
                fs_rng_g.enforce_squeeze_128_bits_nonnative_field_elements(
                    cs.ns(|| "squeeze 128 bits given bytes absorb"), 1
                ).unwrap()[0].get_value().unwrap(),
            )
        );

        if !cs.is_satisfied(){
            println!("{:?}", cs.which_is_unsatisfied());
        }

        assert!(cs.is_satisfied());
    }

    pub(crate) fn test_gadget_squeeze_consistency<
        F: PrimeField,
        ConstraintF: PrimeField,
        FSG: FiatShamirRngGadget<F, ConstraintF>
    >()
    {
        let mut cs = TestConstraintSystem::<ConstraintF>::new();
        let rng = &mut rand::thread_rng();

        let input_size = 5;
        let to_squeeze_max = 5;

        let inputs = vec![ConstraintF::rand(rng); input_size];

        // Alloc native inputs
        let inputs_gs = Vec::<FpGadget<ConstraintF>>::alloc(
            cs.ns(|| "alloc native inputs"),
            || Ok(inputs)
        ).unwrap();


        let mut fs_rng = FSG::new(cs.ns(|| "new FS RNG")).unwrap();
        fs_rng.enforce_absorb_native_field_elements(cs.ns(|| "absorb"), &inputs_gs).unwrap();

        for i in 1..=to_squeeze_max {

            let native_outputs = fs_rng.enforce_squeeze_native_field_elements(
                cs.ns(|| format!("squeeze native {}", i)),
                i
            ).unwrap();
            let non_native_outputs = fs_rng.enforce_squeeze_nonnative_field_elements(
                cs.ns(|| format!("squeeze non-native {}", i)),
                i
            ).unwrap();
            let non_native_128_bits_outputs = fs_rng.enforce_squeeze_128_bits_nonnative_field_elements(
                cs.ns(|| format!("squeeze 128 bits non-native {}", i)),
                i
            ).unwrap();

            // Test squeezing of correct number of field elements
            assert_eq!(i, native_outputs.len());
            assert_eq!(i, non_native_outputs.len());
            assert_eq!(i, non_native_128_bits_outputs.len());
        }

        if !cs.is_satisfied(){
            println!("{:?}", cs.which_is_unsatisfied());
        }

        assert!(cs.is_satisfied());
    }
}