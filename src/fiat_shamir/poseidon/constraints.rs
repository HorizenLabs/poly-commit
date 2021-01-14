use crate::fiat_shamir::constraints::FiatShamirRngGadget;
use algebra::{
    PrimeField, FpParameters
};
use primitives::{PoseidonHash, PoseidonParameters, PoseidonSBox};
use r1cs_crypto::{SBoxGadget, PoseidonHashGadget, AlgebraicSpongeGadget};
use r1cs_core::{ConstraintSystem, SynthesisError};
use r1cs_std::{
    fields::fp::FpGadget,
    to_field_gadget_vec::ToConstraintFieldGadget,
    bits::{
        uint8::UInt8, boolean::Boolean, FromBitsGadget, ToBitsGadget,
    },
};

impl<F, ConstraintF, P, SB, SBG> FiatShamirRngGadget<F, ConstraintF, PoseidonHash<ConstraintF, P, SB>> for PoseidonHashGadget<ConstraintF, P, SB, SBG>
    where
        F: PrimeField,
        ConstraintF: PrimeField,
        P: PoseidonParameters<Fr = ConstraintF>,
        SB: PoseidonSBox<P>,
        SBG: SBoxGadget<ConstraintF, SB>
{
    fn new<CS: ConstraintSystem<ConstraintF>>(cs: CS) -> Result<Self, SynthesisError> {
        <Self as AlgebraicSpongeGadget<PoseidonHash<ConstraintF, P, SB>, ConstraintF>>::new(cs)
    }

    /*fn enforce_absorb_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        mut cs: CS,
        elems: &[F]
    ) -> Result<(), SynthesisError>
    {
        // Serialize elems to bits
        let elems_bits: Vec<bool> = elems
            .iter()
            .flat_map(|fe|{ fe.write_bits() })
            .collect();

        // Allocate bits as witnesses
        let elems_bits_gadget = Vec::<Boolean>::alloc(
            cs.ns(|| "alloc non-native bits"),
            || Ok(elems_bits)
        )?;
        // PROBLEM: NOT BINDED TO ANYTHING !!

        // Enforce packing (safely) bits into native field element gadgets
        let mut native_fe_gadgets = Vec::new();
        for (i, bits) in elems_bits_gadget.chunks(ConstraintF::Params::CAPACITY as usize).enumerate() {
            let fe_g = FpGadget::<ConstraintF>::from_bits(
                cs.ns(|| format!("pack into native fe {}", i)),
                bits
            )?;
            native_fe_gadgets.push(fe_g);
        }

        // Enforce absorbing of native field element gadgets
        self.enforce_absorb(cs.ns(|| "absorb new native fes"), native_fe_gadgets.as_slice())?;

        Ok(())
    }*/

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

    fn enforce_squeeze_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        mut cs: CS,
        num: usize
    ) -> Result<Vec<Vec<Boolean>>, SynthesisError> {
        // Compute number of native field elements we need in order to satisfy the request of
        // num non-native field elements as output
        let nonnative_bits = F::Params::CAPACITY as usize;
        let required_bits = num * nonnative_bits;
        let required_native_fes = {
            let num = (required_bits/ConstraintF::Params::CAPACITY as usize) as f64;
            if num == 0.0 { 1 } else { num.ceil() as usize }
        };

        // Enforce squeezing of required number of native field elements
        let native_squeeze_gs = self.enforce_squeeze(
            cs.ns(|| "squeeze_required native_fes"),
            required_native_fes
        )?;

        // Enforce serialization to bits
        let mut native_bits_gs = Vec::new();
        for (i, fe_g) in native_squeeze_gs.into_iter().enumerate() {
            //TODO: Do we need range proofs ?
            let mut fe_g_bits = fe_g.to_bits(cs.ns(|| format!("fe_g_{}", i)))?;
            native_bits_gs.append(&mut fe_g_bits);
        }

        // Return bits: each F::Params::CAPACITY bits chunk is a non-native field element
        Ok(native_bits_gs.chunks(F::Params::CAPACITY as usize).take(num).map(|bits| {
            let mut bv = bits.to_vec();
            bv.insert(0, Boolean::constant(false));
            bv
        }).collect())
    }

    fn enforce_squeeze_native_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        cs: CS,
        num: usize
    ) -> Result<Vec<FpGadget<ConstraintF>>, SynthesisError> {
        self.enforce_squeeze(cs, num)
    }

    fn enforce_squeeze_128_bits_nonnative_field_elements<CS: ConstraintSystem<ConstraintF>>(
        &mut self,
        mut cs: CS,
        num: usize
    ) -> Result<Vec<Vec<Boolean>>, SynthesisError> {
        // Compute number of challenges we can extract from a single field element
        let modulus_bits = F::Params::MODULUS_BITS as usize;
        assert!(modulus_bits > 128);
        let challenges_per_fe = modulus_bits/128;

        // Compute number of field elements we need in order to provide the requested 'num'
        // 128 bits field elements
        let to_squeeze = {
            let num = (num/challenges_per_fe) as f64;
            if num == 0.0 { 1 } else { num.ceil() as usize }
        };

        // Enforce squeezing the required number of field elements
        let outputs_bits =
            <Self as FiatShamirRngGadget<F, ConstraintF, PoseidonHash<ConstraintF, P, SB>>>::enforce_squeeze_nonnative_field_elements(
                self,
                cs.ns(|| "squeeze_required non_native_fes"),
                to_squeeze
            )?.into_iter().flatten().collect::<Vec<Boolean>>();

        // Take the required amount of 128 bits chunks and read (safely) a non-native field
        // element out of each of them.
        Ok(outputs_bits.chunks(128).take(num).map(|bits| {
            let mut zeros = vec![Boolean::constant(false); modulus_bits - 128];
            let mut bv = bits.to_vec();
            zeros.append(&mut bv);
            zeros
        }).collect())
    }
}

#[cfg(test)]
mod test {

    use algebra::UniformRand;
    use crate::fiat_shamir::constraints::test::test_native_result;
    use rand::thread_rng;
    use rand_core::RngCore;

    #[test]
    fn test_native_poseidon_fs_gadget_bn_382() {

        use algebra::fields::bn_382::{
            fq::Fq, fr::Fr
        };
        use primitives::crh::poseidon::parameters::bn382::BN382FrPoseidonHash;
        use r1cs_crypto::crh::poseidon::bn382::BN382FrPoseidonHashGadget;

        let rng = &mut thread_rng();

        for _ in 0..100 {
            let native_inputs = vec![Fr::rand(rng); 5];
            let mut byte_inputs = vec![0u8; 100];
            rng.fill_bytes(&mut byte_inputs);

            test_native_result::<Fq, Fr, BN382FrPoseidonHash, BN382FrPoseidonHashGadget>(
                native_inputs, &byte_inputs
            );
        }
    }
}