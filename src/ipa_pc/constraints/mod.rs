mod data_structures;
pub use self::data_structures::*;
use algebra::{PrimeField, AffineCurve, ProjectiveCurve, UniformRand, Group};
use r1cs_std::groups::GroupGadget;
use crate::fiat_shamir::constraints::FiatShamirRngGadget;
use r1cs_core::{ToConstraintField, ConstraintSystem, SynthesisError};
use crate::fiat_shamir::FiatShamirRng;
use crate::constraints::{PolynomialCommitmentGadget, QuerySetGadget, EvaluationsGadget, LinearCombinationGadget, BatchLCProofGadget, LabeledPointGadget};
use crate::ipa_pc::{InnerProductArgPC, Commitment};
use r1cs_std::to_field_gadget_vec::ToConstraintFieldGadget;
use r1cs_std::fields::FieldGadget;
use r1cs_std::fields::fp::FpGadget;
use r1cs_std::fields::nonnative::nonnative_field_gadget::NonNativeFieldGadget;
use r1cs_std::ToBitsGadget;
use std::marker::PhantomData;
use std::collections::{
    BTreeMap, BTreeSet
};
use r1cs_std::alloc::ConstantGadget;
use r1cs_std::bits::uint8::UInt8;
use r1cs_std::Assignment;

#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
)]
/// R1CS gadget able to verify IPA polynomial commitments.
pub struct InnerProductArgPCGadget<
    F: PrimeField,
    ConstraintF: PrimeField,
    G: AffineCurve<BaseField = ConstraintF, ScalarField = F> + ToConstraintField<ConstraintF>,
    GG: GroupGadget<G::Projective, ConstraintF>,
    FSG: FiatShamirRngGadget<F, ConstraintF>
> {
    #[doc(hidden)]
    _base_field:    PhantomData<G>,
    #[doc(hidden)]
    _scalar_field:  PhantomData<GG>,
    #[doc(hidden)]
    _group:         PhantomData<G>,
    #[doc(hidden)]
    _group_gadget:  PhantomData<GG>,
    #[doc(hidden)]
    _fs_gadget:     PhantomData<FSG>,
}

impl<F, ConstraintF, G, GG, FSG> InnerProductArgPCGadget<F, ConstraintF, G, GG, FSG>
    where
        F: PrimeField,
        ConstraintF: PrimeField,
        G: AffineCurve<BaseField = ConstraintF, ScalarField = F> + ToConstraintField<ConstraintF>,
        GG: GroupGadget<G::Projective, ConstraintF, Value = G::Projective> + ToConstraintFieldGadget<ConstraintF, FieldGadget = FpGadget<ConstraintF>>,
        FSG: FiatShamirRngGadget<F, ConstraintF>
{
    /// Evaluate the succinct_check_polynomial at a point, starting from the challenges
    fn evaluate_succinct_check_polynomial_from_challenges<CS: ConstraintSystem<ConstraintF>>(
        mut cs:       CS,
        challenges:   &[NonNativeFieldGadget<F, ConstraintF>],
        point:        &NonNativeFieldGadget<F, ConstraintF>,
    ) -> Result<NonNativeFieldGadget<F, ConstraintF>, SynthesisError>
    {
        let log_d = challenges.len();
        let one = NonNativeFieldGadget::<F, ConstraintF>::one(
            cs.ns(|| "alloc one")
        )?;
        let mut product = one.clone();

        for (i, challenge) in challenges.iter().enumerate() {
            let i = i + 1;

            let mut elem_degree_bits = UInt8::constant((1 << (log_d - i)) as u8).into_bits_le();
            elem_degree_bits.reverse();

            let elem = point.pow(
                cs.ns(|| format!("point^elem_{}", i)),
                elem_degree_bits.as_slice()
            )?;
            product = elem
                .mul(cs.ns(|| format!("(elem * challenge)_{}", i)), &challenge)?
                .add(cs.ns(|| format!("(one + elem * challenge)_{}", i)), &one)?
                .add(cs.ns(|| format!("product *= (one + elem * challenge)_{}", i)), &product)?;
        }

        Ok(product)
    }

    /// Enforce succinct_verification and returns the bullet challenges
    pub(crate) fn succinct_check<CS: ConstraintSystem<ConstraintF>>(
        mut cs:           CS,
        verification_key: &PreparedVerifierKeyGadget<G, GG>,
        commitments:      &[LabeledCommitmentGadget<G, GG>],
        point:            &NonNativeFieldGadget<F, ConstraintF>,
        values:           Vec<NonNativeFieldGadget<F, ConstraintF>>,
        proof:            &ProofGadget<G, GG>,
        ro:               &mut FSG,
    ) -> Result<Vec<NonNativeFieldGadget<F, ConstraintF>>, SynthesisError>
    {
        let check_time = start_timer!(|| "Succinct checking");

        // Random shift to avoid exceptional cases if add is incomplete.
        // With overwhelming probability the circuit will be satisfiable,
        // otherwise the prover can sample another shift by re-running
        // the proof creation.
        let shift = GG::alloc(
            cs.ns(|| "alloc random shift for combined commitment proj"),
            || {
                let mut rng = rand_core::OsRng::default();
                Ok(loop {
                    let r = G::Projective::rand(&mut rng);
                    if !r.into_affine().is_zero() { break(r) }
                })
            }
        )?;

        let mut combined_commitment = shift.clone();

        let mut combined_v = NonNativeFieldGadget::zero(cs.ns(|| "alloc combined v"))?;

        //  TODO: Instead of squeezing new challenge each time, we can use the same one and compute
        //        sum_i (ci * chal^i) in an optimized way using Horner's method (avoiding to
        //        compute the powers of lambda).
        let mut cur_challenge = ro.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(
            cs.ns(|| "squeeze first batching chal"),
            1
        )?;

        for (i, (labeled_commitment, value)) in commitments.iter().zip(values.iter()).enumerate() {
            combined_v = cur_challenge.0[0]
                .mul(cs.ns(|| format!("(cur_challenge * value)_{}", i)), value)?
                .add(cs.ns(|| format!("combined_v + (cur_challenge * value)_{}", i)), &combined_v)?;
            let commitment = &labeled_commitment.commitment;

            combined_commitment = commitment.comm.mul_bits(
                cs.ns(|| format!("combined_comm += (comm * chal)_{}", i)),
                &combined_commitment,
                cur_challenge.1[0].iter(),
            )?;

            let degree_bound = labeled_commitment.degree_bound.as_ref();
            assert_eq!(degree_bound.is_some(), commitment.shifted_comm.is_some());

            if let Some(degree_bound) = degree_bound {
                cur_challenge = ro.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(
                    cs.ns(|| format!("squeeze batching chal for degree bound {}", i)),
                    1
                )?;

                // Exponent = supported_degree - degree_bound
                let exponent = FpGadget::<ConstraintF>::from_value(
                    cs.ns(|| "hardcode supported vk degree"),
                    &ConstraintF::from((verification_key.comm_key.len() - 1) as u32) // Will never be bigger than 2^32
                ).sub(cs.ns(|| "exponent = supported_degree - degree_bound"), degree_bound)?;

                // exponent to bits
                let mut exponent_bits = exponent.to_bits_with_length_restriction(
                    cs.ns(|| "exponent to bits strict"),
                    ConstraintF::size_in_bits() - 32
                )?;
                exponent_bits.reverse();

                // compute shift
                let shift = point.pow(
                    cs.ns(|| "point^(supported_degree - degree_bound"),
                    exponent_bits.as_slice()
                )?;

                combined_v = cur_challenge.0[0]
                    .mul(cs.ns(|| format!("(cur_challenge * value)_deg_bound_{}", i)), value)?
                    .mul(cs.ns(|| format!("(cur_challenge * value * shift)_deg_bound_{}", i)), &shift)?
                    .add(cs.ns(|| format!("combined_v + (cur_challenge * value * shift)_deg_bound_{}", i)), &combined_v)?;

                combined_commitment = commitment.shifted_comm.as_ref().unwrap().mul_bits(
                    cs.ns(|| format!("combined_comm += (shifted_comm * chal)_{}", i)),
                    &combined_commitment,
                    cur_challenge.1[0].iter(),
                )?;
            }

            cur_challenge = ro.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(
                cs.ns(|| format!("squeeze batching chal {}", i)),
                1
            )?;
        }

        assert_eq!(proof.hiding_comm.is_some(), proof.rand.is_some());
        if proof.hiding_comm.is_some() {
            let hiding_comm = proof.hiding_comm.as_ref().unwrap();
            let rand = proof.rand.as_ref().unwrap();

            //TODO: Range proof here ?
            let rand_bits = rand.to_bits_strict(cs.ns(|| "rand to bits strict"))?;

            let hiding_challenge = {
                ro.enforce_absorb_nonnative_field_elements(
                    cs.ns(|| "absorb nonnative for hiding_challenge"),
                    &[point.clone(), combined_v.clone()]
                )?;
                ro.enforce_absorb_native_field_elements(
                    cs.ns(|| "absorb native for hiding_challenge"),
                    &[combined_commitment.clone(), hiding_comm.clone()]
                )?;
                ro.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(
                    cs.ns(|| "squeeze hiding_challenge"),
                    1
                )
            }?;
            
            combined_commitment = hiding_comm.mul_bits(
                cs.ns(|| "combined_comm += hiding_comm * hiding_chal"),
                &combined_commitment,
                hiding_challenge.1[0].iter()
            )?;
            
            let neg_vk_s = verification_key.s.negate(cs.ns(|| "-vk.s"))?;

            combined_commitment = neg_vk_s.mul_bits(
                cs.ns(|| "combined_comm += (hiding_comm * hiding_chal - vk.s * rand)"),
                &combined_commitment,
                rand_bits.iter().rev()
            )?;
        }

        // Challenge for each round
        let mut round_challenges = Vec::new();
        let round_challenge = {
            ro.enforce_absorb_native_field_elements(
                cs.ns(|| "absorb native for first round chal"),
                &[combined_commitment.clone()]
            )?;
            ro.enforce_absorb_nonnative_field_elements(
                cs.ns(|| "absorb nonnative for first round chal"),
                &[point.clone(), combined_v.clone()]
            )?;
            ro.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(
                cs.ns(|| "squeeze first round chal"),
                1
            )
        }?;
        
        let h_prime = {
            // Random shift to avoid exceptional cases if add is incomplete.
            // With overwhelming probability the circuit will be satisfiable,
            // otherwise the prover can sample another shift by re-running
            // the proof creation.
            let shift = GG::alloc(
                cs.ns(|| "alloc random shift for h_prime"),
                || {
                    let mut rng = rand_core::OsRng::default();
                    Ok(loop {
                        let r = G::Projective::rand(&mut rng);
                        if !r.into_affine().is_zero() { break(r) }
                    })
                }
            )?;
            
            let result = shift.clone();
            
            let shifted_h_prime = verification_key.h.mul_bits(
                cs.ns(|| "shifted_h_prime"),
                &result,
                round_challenge.1[0].iter()
            )?;
            
            shifted_h_prime.sub(cs.ns(|| "h_prime"), &shift)
        }?;

        // Compute combined_commitment by subtracting shift
        combined_commitment = combined_commitment.sub(cs.ns(|| "combined_commitment"), &shift)?;

        // Compute combined_v bits
        //TODO: Range proof here ?
        let combined_v_bits = combined_v.to_bits_strict(cs.ns(|| "combined_v to bits strict"))?;

        let mut round_commitment = h_prime.mul_bits(
            cs.ns(|| "round_commitment = combined_commitment + h_prime * combined_v"),
            &combined_commitment,
            combined_v_bits.iter().rev()
        )?;

        let l_iter = proof.l_vec.iter();
        let r_iter = proof.r_vec.iter();

        let round_shift = GG::alloc(
            cs.ns(|| format!("alloc random shift to enforce round constraints")),
            || {
                let mut rng = rand_core::OsRng::default();
                Ok(loop {
                    let r = G::Projective::rand(&mut rng);
                    if !r.into_affine().is_zero() { break(r) }
                })
            }
        )?;

        for (i, (l, r)) in l_iter.zip(r_iter).enumerate() {
            let round_challenge = {
                ro.enforce_absorb_nonnative_field_elements(
                    cs.ns(|| format!("absorb nonnative for round_chal_{}", i)),
                    &[round_challenge.0[0].clone()]
                )?;
                ro.enforce_absorb_native_field_elements(
                    cs.ns(|| format!("absorb native for round_chal_{}", i)),
                    &[l.clone(), r.clone()]
                )?;
                ro.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(
                    cs.ns(|| format!("squeeze round_chal_{}", i)),
                    1
                )
            }?;

            // We need to compute l ^ round_chal_inv and add it to the round_commitment.
            // Instead of explicitly enforcing the computation of the inverse in the circuit,
            // we let the prover choose a new point temp and we enforce that temp ^ round_chal = l.
            // The assignment is satisfied iff temp = l^round_chal_inv.
            {
                let temp = GG::alloc(
                    cs.ns(|| format!("alloc (chal_inv * l)_{}", i)),
                    || {
                        let l_val = l.get_value().get()?;
                        let round_chal_inv_val = round_challenge.0[0].get_value().get()?.inverse().get()?;
                        Ok(l_val.mul(&round_chal_inv_val))
                    }
                )?;

                temp.mul_bits(
                    cs.ns(|| format!("(chal * temp)_{}", i)),
                    &round_shift,
                    round_challenge.1[0].clone().iter()
                )?
                    .sub(cs.ns(|| format!("subtract_shift_{}", i)), &round_shift)?
                    .enforce_equal(cs.ns(|| format!("(chal * temp == l)_{}", i)), &l)?;

                round_commitment = round_commitment.add(
                    cs.ns(|| format!("round_commitment += ((l* 1_over_round_chal))_{}", i)),
                    &temp
                )?;
            }

            round_commitment = r.mul_bits(
                cs.ns(|| format!("round_commitment += ((l* 1_over_round_chal) + (r * round_chal))_{}", i)),
                &round_commitment,
                round_challenge.1[0].clone().iter()
            )?;

            round_challenges.push(round_challenge);
        }

        let xi_s = round_challenges.into_iter().map(|chal| chal.0[0].clone()).collect::<Vec<_>>();
        let v_prime = Self::evaluate_succinct_check_polynomial_from_challenges(
            cs.ns(|| "eval succinct check poly at point"),
            xi_s.as_slice(),
            &point
        )?.mul(cs.ns(|| "succinct_poly(point) * proof.c"), &proof.c)?;

        let check_commitment_elem = Self::cm_commit(
            cs.ns(|| "compute final comm"),
            &[proof.final_comm_key.clone(), h_prime],
            &[proof.c.clone(), v_prime],
            None,
            None,
        )?;

        check_commitment_elem.enforce_equal(cs.ns(|| "check final comm"), &round_commitment)?;

        end_timer!(check_time);
        Ok(xi_s)
    }

    /// Create a Pedersen commitment to `scalars` using the commitment key `comm_key`.
    /// Optionally, randomize the commitment using `hiding_generator` and `randomizer`.
    fn cm_commit<CS: ConstraintSystem<ConstraintF>>(
        mut cs:           CS,
        comm_key:         &[GG],
        scalars:          &[NonNativeFieldGadget<F, ConstraintF>],
        hiding_generator: Option<GG>,
        randomizer:       Option<NonNativeFieldGadget<F, ConstraintF>>,
    ) -> Result<GG, SynthesisError>
    {
        // Random shift to avoid exceptional cases if add is incomplete.
        // With overwhelming probability the circuit will be satisfiable,
        // otherwise the prover can sample another shift by re-running
        // the proof creation.
        let shift = GG::alloc(cs.ns(|| "alloc random shift for MSM result"), || {
            let mut rng = rand_core::OsRng::default();
            Ok(loop {
                let r = G::Projective::rand(&mut rng);
                if !r.into_affine().is_zero() { break(r) }
            })
        })?;

        let mut comm = shift.clone();

        // Variable base MSM
        for (i, (base, scalar)) in comm_key.iter().zip(scalars.iter()).enumerate() {

            //TODO: Range proof here ?
            let scalar_bits = scalar.to_bits_strict(
                cs.ns(|| format!("scalar_{} to bits strict", i))
            )?;

            comm = base.mul_bits(
                cs.ns(|| format!("comm += (base_{} * scalar_{})", i, i)),
                &comm,
                scalar_bits.iter().rev()
            )?;
        }

        // Subtract shift from commitment
        comm = comm.sub(cs.ns(|| "subtract shift"), &shift)?;

        // Randomize if needed
        if randomizer.is_some() {
            assert!(hiding_generator.is_some());

            //TODO: Range proof here ?
            let randomizer_bits = randomizer.unwrap().to_bits_strict(
                cs.ns(|| "randomizer to bits strict"),
            )?;

            comm = hiding_generator.unwrap().mul_bits(
                cs.ns(|| "comm += (hiding_generator * randomizer)"),
                &comm,
                randomizer_bits.iter().rev()
            )?;
        }

        Ok(comm)
    }
}

impl<F, ConstraintF, G, GG, FS, FSG> PolynomialCommitmentGadget<G, InnerProductArgPC<ConstraintF, G, FS>>
for InnerProductArgPCGadget<F, ConstraintF, G, GG, FSG>
    where
        F: PrimeField,
        ConstraintF: PrimeField,
        G: AffineCurve<BaseField = ConstraintF, ScalarField = F> + ToConstraintField<ConstraintF>,
        GG: GroupGadget<G::Projective, ConstraintF, Value = G::Projective> + ToConstraintFieldGadget<ConstraintF, FieldGadget = FpGadget<ConstraintF>>,
        FS: FiatShamirRng<F, ConstraintF>,
        FSG: FiatShamirRngGadget<F, ConstraintF>
{
    type VerifierKeyGadget = VerifierKeyGadget<G, GG>;
    type PreparedVerifierKeyGadget = PreparedVerifierKeyGadget<G, GG>;
    type CommitmentGadget = CommitmentGadget<G, GG>;
    type PreparedCommitmentGadget = PreparedCommitmentGadget<G, GG>;
    type LabeledCommitmentGadget = LabeledCommitmentGadget<G, GG>;
    type PreparedLabeledCommitmentGadget = PreparedLabeledCommitmentGadget<G, GG>;
    type RandomOracleGadget = FSG;
    type ProofGadget = ProofGadget<G, GG>;
    type BatchProofGadget = BatchProofGadget<G, GG>;

    fn prepared_batch_check_individual_opening_challenges<CS: ConstraintSystem<ConstraintF>>(
        mut cs: CS,
        verification_key: &Self::PreparedVerifierKeyGadget,
        commitments:      &[Self::PreparedLabeledCommitmentGadget],
        query_set:        &QuerySetGadget<F, ConstraintF>,
        evaluations:      &EvaluationsGadget<F, ConstraintF>,
        proof:            &Self::BatchProofGadget,
        fs_rng:           &mut Self::RandomOracleGadget
    ) -> Result<(), SynthesisError>
    {
        // Put commitments into a Map<PolyLabel, Commitment>
        let commitments: BTreeMap<_, _> = commitments.into_iter().map(|c| (c.label.clone(), c.clone())).collect();
        let mut query_to_labels_map = BTreeMap::new();

        // Re-organize QuerySet into a Map<PolyLabel, Set<LabeledPoint>>, associating to
        // each poly, the list of points in which it must be evaluated.
        for (poly_label, labeled_point) in query_set.0.iter() {
            let labels = query_to_labels_map
                .entry(labeled_point.0.clone())
                .or_insert((labeled_point.1.clone(), BTreeSet::new()));
            labels.1.insert(poly_label);
        }

        assert_eq!(proof.0.len(), query_to_labels_map.len());

        for (i, ((_point_label, (point, labels)), p)) in query_to_labels_map.into_iter().zip(proof.0.iter()).enumerate() {
            let lc_time =
                start_timer!(|| format!("Randomly combining {} commitments", labels.len()));
            let mut comms = Vec::new();
            let mut vals = Vec::new();
            for label in labels.into_iter() {

                //TODO: Add error to get failed commitment label
                let commitment = commitments.get(label).ok_or(SynthesisError::AssignmentMissing)?;

                //TODO: Add error to get failed evaluation label
                let v_i = evaluations.0
                    .get(&LabeledPointGadget(label.clone(), point.clone()))

                    .ok_or(SynthesisError::AssignmentMissing)?;

                comms.push(commitment.clone());
                vals.push(v_i.clone());
            }

            // Succinct verifier, return the check_poly
            let _ = Self::succinct_check(
                cs.ns(|| format!("succinct check {}", i)),
                verification_key,
                comms.as_slice(),
                &point,
                vals,
                &p,
                fs_rng
            )?;

            end_timer!(lc_time);
        }

        Ok(())
    }

    fn prepared_check_combinations<CS: ConstraintSystem<ConstraintF>>(
        _cs: CS,
        _prepared_verification_key: &Self::PreparedVerifierKeyGadget,
        _linear_combinations:       &[LinearCombinationGadget<F, ConstraintF>],
        _prepared_commitments:      &[Self::PreparedLabeledCommitmentGadget],
        _query_set:                 &QuerySetGadget<F, ConstraintF>,
        _evaluations:               &EvaluationsGadget<F, ConstraintF>,
        _proof:                     &BatchLCProofGadget<G, InnerProductArgPC<ConstraintF, G, FS>, Self>,
        _fs_rng:                    &mut Self::RandomOracleGadget
    ) -> Result<(), SynthesisError>
    {
        unimplemented!()
    }

    fn create_labeled_commitment(
        label:        String,
        commitment:   Self::CommitmentGadget,
        degree_bound: Option<FpGadget<ConstraintF>>,
    ) -> Self::LabeledCommitmentGadget
    {
        LabeledCommitmentGadget::<G, GG>{
            label,
            commitment,
            degree_bound
        }
    }

    fn create_prepared_labeled_commitment(
        label:        String,
        commitment:   Self::PreparedCommitmentGadget,
        degree_bound: Option<FpGadget<ConstraintF>>,
    ) -> Self::PreparedLabeledCommitmentGadget
    {
        LabeledCommitmentGadget::<G, GG>{
            label,
            commitment,
            degree_bound
        }
    }

    fn verify_polynomial_commitment_from_lagrange_representation<CS: ConstraintSystem<ConstraintF>>(
        mut cs:              CS,
        expected_comm:       &Self::CommitmentGadget,
        lagrange_poly_comms: &[Commitment<G>],
        poly_coords:         &[NonNativeFieldGadget<F, ConstraintF>],
    ) -> Result<(), SynthesisError>
    {
        assert_eq!(poly_coords.len(), lagrange_poly_comms.len());

        // Get the bits from the non native field gadget
        let poly_coords_bits = poly_coords.iter().enumerate().map(|(i, poly_coord)| {
            //TODO: Is range proof really needed here ?
            let mut bits = poly_coord.to_bits_strict(cs.ns(|| format!("poly coord {} to bits strict", i)))?;
            bits.reverse();
            Ok(bits)
        }).collect::<Result<Vec<_>, SynthesisError>>()?;

        // Random shift to avoid exceptional cases if add is incomplete.
        // With overwhelming probability the circuit will be satisfiable,
        // otherwise the prover can sample another shift by re-running
        // the proof creation.
        let shift = GG::alloc(cs.ns(|| "alloc random shift"), || {
            let mut rng = rand_core::OsRng::default();
            Ok(loop {
                let r = G::Projective::rand(&mut rng);
                if !r.into_affine().is_zero() { break(r) }
            })
        })?;

        // Fixed Base MSM
        // WARNING: If the addition for G is incomplete and one of the bit sequences is
        // all 0s, this will result in a crash.
        let mut result = shift.clone();
        for (i, bits) in poly_coords_bits.into_iter().enumerate() {
            result = GG::mul_bits_fixed_base(
                &lagrange_poly_comms[i].comm.into_projective(),
                cs.ns(|| format!("x_{} * COMM(L_{})", i, i)),
                &result,
                &bits
            )?;
        }
        result = result.sub(cs.ns(|| format!("subtract shift")), &shift)?;

        result.enforce_equal(cs.ns(|| "actual_comm == expected_comm"), &expected_comm.comm)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_camel_case_types)]

    use super::InnerProductArgPC;

    use algebra::{
        fields::tweedle::{
            Fq, Fr
        },
        curves::tweedle::dee::{
            Affine, TweedledeeParameters as AffineParameters
        }
    };
    use blake2::Blake2s;
    use crate::ipa_pc::constraints::InnerProductArgPCGadget;
    use r1cs_std::fields::fp::FpGadget;
    use primitives::TweedleFqPoseidonSponge;
    use r1cs_crypto::TweedleFqPoseidonSpongeGadget;
    use r1cs_std::groups::curves::short_weierstrass::short_weierstrass_jacobian::AffineGadget as GroupGadget;


    type PC<F, G, FS> = InnerProductArgPC<F, G, FS>;
    type PC_GADGET<F, ConstraintF, G, GG, FSG> = InnerProductArgPCGadget<F, ConstraintF, G, GG, FSG>;

    type AffineGadget = GroupGadget<AffineParameters, Fq, FpGadget<Fq>>;
    type PC_TWEEDLE = PC<Fq, Affine, TweedleFqPoseidonSponge>;
    type PC_TWEEDLE_GADGET = PC_GADGET<Fr, Fq, Affine, AffineGadget, TweedleFqPoseidonSpongeGadget>;
    
    #[test]
    fn constant_poly_test() {
        use crate::constraints::test::*;
        constant_poly_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>().expect("test failed for tweedle_dum-blake2s");
    }

    #[test]
    fn single_poly_test() {
        use crate::constraints::test::*;
        single_poly_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>().expect("test failed for tweedle_dum-blake2s");
    }

    #[test]
    fn quadratic_poly_degree_bound_multiple_queries_test() {
        use crate::constraints::test::*;
        quadratic_poly_degree_bound_multiple_queries_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>()
            .expect("test failed for tweedle_dum-blake2s");
    }

    #[test]
    fn linear_poly_degree_bound_test() {
        use crate::constraints::test::*;
        linear_poly_degree_bound_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>()
            .expect("test failed for tweedle_dum-blake2s");
    }

    #[test]
    fn single_poly_degree_bound_test() {
        use crate::constraints::test::*;
        single_poly_degree_bound_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>()
            .expect("test failed for tweedle_dum-blake2s");
    }

    #[test]
    fn single_poly_degree_bound_multiple_queries_test() {
        use crate::constraints::test::*;

        single_poly_degree_bound_multiple_queries_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>()
            .expect("test failed for tweedle_dum-blake2s");
    }

    #[test]
    fn two_polys_degree_bound_single_query_test() {
        use crate::constraints::test::*;

        two_polys_degree_bound_single_query_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>()
            .expect("test failed for tweedle_dum-blake2s");
    }

    #[test]
    fn full_end_to_end_test() {
        use crate::constraints::test::*;

        full_end_to_end_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>().expect("test failed for tweedle_dum-blake2s");
        println!("Finished tweedle_dum-blake2s");
    }

    #[test]
    #[should_panic]
    fn bad_degree_bound_test() {
        use crate::constraints::test::*;

        bad_degree_bound_test::<_, PC_TWEEDLE, PC_TWEEDLE_GADGET, Blake2s>().expect("test failed for tweedle_dum-blake2s");
        println!("Finished tweedle_dum-blake2s");
    }
}