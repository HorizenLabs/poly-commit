use crate::fiat_shamir::FiatShamirRng;
use algebra::PrimeField;
use r1cs_core::{ConstraintSystem, SynthesisError};
use r1cs_std::{
    fields::fp::FpGadget,
    to_field_gadget_vec::ToConstraintFieldGadget
};
use r1cs_std::bits::{
    uint8::UInt8, boolean::Boolean
};

// TODO: In the future, we will have a non-native field element representation in the SNARK circuit
//       (a gadget) that handles all the boiler plate and provides more flexibility. This is already
//       implemented in arkworks/nonnative.

/// the trait for Fiat-Shamir RNG
pub trait FiatShamirRngGadget<F: PrimeField, ConstraintF: PrimeField, FS: FiatShamirRng<F, ConstraintF>>: Sized {

    /// initialize the RNG
    fn new<CS: ConstraintSystem<ConstraintF>>(cs: CS) -> Result<Self, SynthesisError>;

    /// take in field elements
    fn enforce_absorb_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        elems: &[F]
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

    /// take out field elements
    fn enforce_squeeze_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<Vec<Vec<Boolean>>, SynthesisError>;

    /// take in field elements
    fn enforce_squeeze_native_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<Vec<FpGadget<ConstraintF>>, SynthesisError>;

    /// take out field elements of 128 bits
    fn enforce_squeeze_128_bits_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<Vec<Vec<Boolean>>, SynthesisError>;
}

#[cfg(test)]
pub(crate) mod test {
    use algebra::PrimeField;
    use crate::fiat_shamir::FiatShamirRng;
    use crate::fiat_shamir::constraints::FiatShamirRngGadget;
    use r1cs_std::test_constraint_system::TestConstraintSystem;
    use r1cs_core::ConstraintSystem;
    use r1cs_std::bits::boolean::Boolean;
    use r1cs_std::fields::fp::FpGadget;
    use r1cs_std::alloc::AllocGadget;
    use r1cs_std::bits::uint8::UInt8;

    pub(crate) fn test_native_result<
        F: PrimeField,
        ConstraintF: PrimeField,
        FS:  FiatShamirRng<F, ConstraintF>,
        FSG: FiatShamirRngGadget<F, ConstraintF, FS>,
    >(
        non_native_inputs: Vec<F>,
        native_inputs: Vec<ConstraintF>,
        byte_inputs: &[u8],
    )
    {
        let gadget_to_primitive = |bit_gadgets: Vec<Boolean>| -> Vec<bool> {
            bit_gadgets.into_iter().map(|bit_gadget| bit_gadget.get_value().unwrap()).collect()
        };

        let mut cs = TestConstraintSystem::<ConstraintF>::new();

        // Non Native inputs
        let mut fs_rng = FS::new();
        fs_rng.absorb_nonnative_field_elements(non_native_inputs.as_slice());

        let mut fs_rng_g = FSG::new(cs.ns(|| "new fs_rng_g for non native inputs")).unwrap();
        fs_rng_g.enforce_absorb_nonnative_field_elements(
            cs.ns(|| "enforce absorb non native field elements"), non_native_inputs.as_slice()
        ).unwrap();
        assert_eq!(
            (
                fs_rng.squeeze_nonnative_field_elements(1)[0].write_bits(),
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0].write_bits(),
            ),
            (
                gadget_to_primitive(fs_rng_g.enforce_squeeze_nonnative_field_elements(
                    cs.ns(||"squeeze non native given non native absorb"), 1
                ).unwrap()[0].clone()),
                fs_rng_g.enforce_squeeze_native_field_elements(
                    cs.ns(|| "squeeze native given non native absorb"), 1
                ).unwrap()[0].value.unwrap(),
                gadget_to_primitive(fs_rng_g.enforce_squeeze_128_bits_nonnative_field_elements(
                    cs.ns(|| "squeeze 128 bits given non native absorb"), 1
                ).unwrap()[0].clone()),
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
                fs_rng.squeeze_nonnative_field_elements(1)[0].write_bits(),
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0].write_bits(),
            ),
            (
                gadget_to_primitive(fs_rng_g.enforce_squeeze_nonnative_field_elements(
                    cs.ns(||"squeeze non native given native absorb"), 1
                ).unwrap()[0].clone()),
                fs_rng_g.enforce_squeeze_native_field_elements(
                    cs.ns(|| "squeeze native given native absorb"), 1
                ).unwrap()[0].value.unwrap(),
                gadget_to_primitive(fs_rng_g.enforce_squeeze_128_bits_nonnative_field_elements(
                        cs.ns(|| "squeeze 128 bits given native absorb"), 1
                ).unwrap()[0].clone()),
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
                fs_rng.squeeze_nonnative_field_elements(1)[0].write_bits(),
                fs_rng.squeeze_native_field_elements(1)[0],
                fs_rng.squeeze_128_bits_nonnative_field_elements(1)[0].write_bits(),
            ),
            (
                gadget_to_primitive(fs_rng_g.enforce_squeeze_nonnative_field_elements(
                    cs.ns(||"squeeze non native given bytes absorb"), 1
                ).unwrap()[0].clone()),
                fs_rng_g.enforce_squeeze_native_field_elements(
                    cs.ns(|| "squeeze native given bytes absorb"), 1
                ).unwrap()[0].value.unwrap(),
                gadget_to_primitive(fs_rng_g.enforce_squeeze_128_bits_nonnative_field_elements(
                    cs.ns(|| "squeeze 128 bits given bytes absorb"), 1
                ).unwrap()[0].clone()),
            )
        );

        if !cs.is_satisfied(){
            println!("{:?}", cs.which_is_unsatisfied());
        }

        assert!(cs.is_satisfied());
    }
}