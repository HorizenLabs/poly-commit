use crate::fiat_shamir::constraints::{FiatShamirRngGadget, push_gadgets_to_sponge, get_gadgets_and_bits_from_sponge};
use algebra::{
    PrimeField, FpParameters
};
use primitives::{PoseidonSponge, PoseidonParameters, PoseidonSBox};
use r1cs_crypto::{SBoxGadget, PoseidonSpongeGadget, AlgebraicSpongeGadget};
use r1cs_core::{ConstraintSystem, SynthesisError};
use r1cs_std::{
    fields::fp::FpGadget,
    to_field_gadget_vec::ToConstraintFieldGadget,
    bits::{
        uint8::UInt8, boolean::Boolean, FromBitsGadget,
    },
};
use r1cs_std::fields::nonnative::nonnative_field_gadget::NonNativeFieldGadget;

impl<F, ConstraintF, P, SB, SBG> FiatShamirRngGadget<F, ConstraintF>
for PoseidonSpongeGadget<ConstraintF, P, SB, SBG>
    where
        F: PrimeField,
        ConstraintF: PrimeField,
        P: PoseidonParameters<Fr = ConstraintF>,
        SB: PoseidonSBox<P>,
        SBG: SBoxGadget<ConstraintF, SB>
{
    fn new<CS: ConstraintSystem<ConstraintF>>(cs: CS) -> Result<Self, SynthesisError> {
        <Self as AlgebraicSpongeGadget<PoseidonSponge<ConstraintF, P, SB>, ConstraintF>>::new(cs)
    }

    fn enforce_absorb_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        elems: &[NonNativeFieldGadget<F, ConstraintF>]
    ) -> Result<(), SynthesisError> {
        push_gadgets_to_sponge::<_, _, PoseidonSponge<ConstraintF, P, SB>, _, _>(cs, self, elems)
    }

    fn enforce_absorb_native_field_elements<
        CS: ConstraintSystem<ConstraintF>,
        T: ToConstraintFieldGadget<ConstraintF, FieldGadget = FpGadget<ConstraintF>>>
    (
        &mut self,
        mut cs: CS,
        elems: &[T]
    )  -> Result<(), SynthesisError>
    {
        let mut input_gs = Vec::new();
        for (i, elem) in elems.into_iter().enumerate() {
            let mut fe_gs = elem.to_field_gadget_elements(cs.ns(|| format!("elem_{}_to_field_gadgets", i)))?;
            input_gs.append(&mut fe_gs);
        }
        self.enforce_absorb(cs.ns(|| "absorb native fes"), input_gs.as_slice())
    }

    fn enforce_absorb_bytes<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        mut cs: CS,
        elems: &[UInt8]
    ) -> Result<(), SynthesisError>
    {
        // Convert [UInt8] to the underlying [Boolean] in big-endian form
        let bytes_to_bool_gs = elems.iter().flat_map(|byte| {
            byte.into_bits_le().iter().rev().cloned().collect::<Vec<Boolean>>()
        }).collect::<Vec<Boolean>>();

        // Enforce packing (safely) bits into native field element gadgets
        let mut native_fe_gadgets = Vec::new();
        for (i, bits) in bytes_to_bool_gs.chunks(ConstraintF::Params::CAPACITY as usize).enumerate() {
            let fe_g = FpGadget::<ConstraintF>::from_bits(
                cs.ns(|| format!("pack into native fe {}", i)),
                bits
            )?;
            native_fe_gadgets.push(fe_g);
        }

        // Enforce absorbing of native field element gadgets
        self.enforce_absorb(cs.ns(|| "absorb native fes"), native_fe_gadgets.as_slice())
    }

    fn enforce_squeeze_nonnative_field_elements_and_bits<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<(Vec<NonNativeFieldGadget<F, ConstraintF>>, Vec<Vec<Boolean>>), SynthesisError> {
        get_gadgets_and_bits_from_sponge(cs, self, num, false)
    }

    fn enforce_squeeze_native_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<Vec<FpGadget<ConstraintF>>, SynthesisError> {
        self.enforce_squeeze(cs, num)
    }

    fn enforce_squeeze_128_bits_nonnative_field_elements_and_bits<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<(Vec<NonNativeFieldGadget<F, ConstraintF>>, Vec<Vec<Boolean>>), SynthesisError> {
        get_gadgets_and_bits_from_sponge(cs, self, num, true)
    }
}

#[cfg(test)]
mod test {
    use algebra::fields::tweedle::{
        fq::Fq, fr::Fr
    };
    use algebra::UniformRand;
    use crate::fiat_shamir::constraints::test::test_native_result;
    use crate::fiat_shamir::constraints::test::test_gadget_squeeze_consistency;
    use rand::thread_rng;
    use rand_core::RngCore;

    #[test]
    fn test_native_poseidon_fs_gadget_tweedle_fr() {

        use primitives::crh::poseidon::parameters::tweedle::TweedleFrPoseidonSponge;
        use r1cs_crypto::crh::poseidon::tweedle::TweedleFrPoseidonSpongeGadget;

        let rng = &mut thread_rng();

        for _ in 0..10 {
            let non_native_inputs = vec![Fq::rand(rng); 5];
            let native_inputs = vec![Fr::rand(rng); 5];
            let mut byte_inputs = vec![0u8; 100];
            rng.fill_bytes(&mut byte_inputs);

            test_native_result::<Fq, Fr, TweedleFrPoseidonSponge, TweedleFrPoseidonSpongeGadget>(
                non_native_inputs, native_inputs, &byte_inputs
            );

            test_gadget_squeeze_consistency::<Fq, Fr, TweedleFrPoseidonSpongeGadget>();
        }
    }

    #[test]
    fn test_native_poseidon_fs_gadget_tweedle_fq() {

        use primitives::crh::poseidon::parameters::tweedle::TweedleFqPoseidonSponge;
        use r1cs_crypto::crh::poseidon::tweedle::TweedleFqPoseidonSpongeGadget;

        let rng = &mut thread_rng();

        for _ in 0..10 {
            let non_native_inputs = vec![Fr::rand(rng); 5];
            let native_inputs = vec![Fq::rand(rng); 5];
            let mut byte_inputs = vec![0u8; 100];
            rng.fill_bytes(&mut byte_inputs);

            test_native_result::<Fr, Fq, TweedleFqPoseidonSponge, TweedleFqPoseidonSpongeGadget>(
                non_native_inputs, native_inputs, &byte_inputs
            );

            test_gadget_squeeze_consistency::<Fr, Fq, TweedleFqPoseidonSpongeGadget>();
        }
    }
}