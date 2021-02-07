mod data_structures;
pub use self::data_structures::*;
use algebra::{PrimeField, AffineCurve, Field, ProjectiveCurve, UniformRand};
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
        GG: GroupGadget<G::Projective, ConstraintF> + ToConstraintFieldGadget<ConstraintF, FieldGadget = FpGadget<ConstraintF>>,
        FSG: FiatShamirRngGadget<F, ConstraintF>
{
    /// Return and enforce the coefficients of the succinct check polynomial
    pub(crate) fn succinct_check<'a, CS: ConstraintSystem<ConstraintF>>(
        _cs:           CS,
        _verification_key: &PreparedVerifierKeyGadget<G, GG>,
        _commitments:      &[LabeledCommitmentGadget<G, GG>],
        _point:            &NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,
        _values:           Vec<NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>>,
        _proof:            &ProofGadget<G, GG>,
        _ro:               &mut FSG,
    ) -> Result<Vec<NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>>, SynthesisError>

    {
        unimplemented!();
    }

    /// Multiply all elements of `coeffs` by `scale`
    #[inline]
    fn scale_poly<CS: ConstraintSystem<ConstraintF>>(
        mut cs: CS,
        coeffs: &mut [NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>],
        scale:  &NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,
    ) -> Result<(), SynthesisError>
    {
        coeffs.iter_mut().enumerate().map(|(i, coeff)| {
            coeff.mul_in_place(cs.ns(|| format!("coeff_{} * scale", i)), scale)?;
            Ok(())
        }).collect::<Result<(), SynthesisError>>()
    }

    /// Add, coefficient-wise, `self_poly` to `other_poly`
    #[inline]
    fn add_polys<CS: ConstraintSystem<ConstraintF>>(
        mut cs:      CS,
        self_poly:   Vec<NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>>,
        other_poly:  Vec<NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>>,
    ) -> Result<Vec<NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>>, SynthesisError>
    {
        let mut result = if self_poly.len() >= other_poly.len() { self_poly } else { other_poly.clone() };

        result.iter_mut().zip(other_poly.iter()).enumerate().map(|(i, (self_coeff, other_coeff))| {
            self_coeff.add_in_place(cs.ns(|| format!("self_coeff_{} + other_coeff_{}", i, i)), &other_coeff)?;
            Ok(())
        }).collect::<Result<(), SynthesisError>>()?;

        Ok(result)
    }

    /// Create a Pedersen commitment to `scalars` using the commitment key `comm_key`.
    /// Optionally, randomize the commitment using `hiding_generator` and `randomizer`.
    fn cm_commit<CS: ConstraintSystem<ConstraintF>>(
        mut cs:           CS,
        comm_key:         &[GG],
        scalars:          &[NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>],
        hiding_generator: Option<GG>,
        randomizer:       Option<NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>>,
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
                cs.ns(|| format!("base_{} * scalar_{}", i, i)),
                &comm,
                scalar_bits.iter()
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
                cs.ns(|| "hiding_generator * randomizer"),
                &comm,
                randomizer_bits.iter()
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
        GG: GroupGadget<G::Projective, ConstraintF> + ToConstraintFieldGadget<ConstraintF, FieldGadget = FpGadget<ConstraintF>>,
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

        // For each evaluation point, perform the multi-poly single-point opening
        // check of the polynomials queried at that point.
        // Additionaly batch the resulting check polynomials (the xi_s) and GFinals in a single
        // dlog hard verification.

        // Batching challenge
        let mut randomizer = fs_rng.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(
            cs.ns(|| "squeeze initial randomizer"), 1
        )?;

        // Batching bullet poly
        let mut combined_check_poly_coeffs = Vec::new();

        // Random shift to avoid exceptional cases if add is incomplete.
        // With overwhelming probability the circuit will be satisfiable,
        // otherwise the prover can sample another shift by re-running
        // the proof creation.
        let shift = GG::alloc(cs.ns(|| "alloc random shift for combined_final_key"), || {
            let mut rng = rand_core::OsRng::default();
            Ok(loop {
                let r = G::Projective::rand(&mut rng);
                if !r.into_affine().is_zero() { break(r) }
            })
        })?;

        // Batching GFinal
        let mut combined_final_key = shift.clone();

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
            let mut check_poly_coeffs = Self::succinct_check(
                cs.ns(|| format!("succinct check {}", i)),
                verification_key,
                comms.as_slice(),
                &point,
                vals,
                &p,
                fs_rng
            )?;

            // check_poly batching
            Self::scale_poly(
                cs.ns(|| format!("scale check poly {}", i)),
                check_poly_coeffs.as_mut_slice(),
                &randomizer.0[0]
            )?;

            if combined_check_poly_coeffs.len() == 0 {
                combined_check_poly_coeffs = check_poly_coeffs;
            } else {
                combined_check_poly_coeffs = Self::add_polys(
                    cs.ns(|| format!("combined_check_poly_{}", i)),
                    combined_check_poly_coeffs,
                    check_poly_coeffs
                )?;
            }

            // GFinal batching
            combined_final_key = p.final_comm_key.mul_bits(
                cs.ns(|| format!("combine G final from proof_{}", i)),
                &combined_final_key,
                randomizer.1[0].iter(),
            )?;

            // Squeeze new batching challenge
            randomizer = fs_rng.enforce_squeeze_128_bits_nonnative_field_elements_and_bits(
                cs.ns(|| format!("squeeze randomizer {}", i)), 1
            )?;

            end_timer!(lc_time);
        }

        // Subtract shift from batched GFinal
        combined_final_key = combined_final_key.sub(cs.ns(|| "subtract shift"), &shift)?;

        let proof_time = start_timer!(|| "Checking batched proof");

        // Dlog hard part
        let final_key = Self::cm_commit(
            cs.ns(|| "compute GFinal"),
            verification_key.comm_key.as_slice(),
            combined_check_poly_coeffs.as_slice(),
            None,
            None,
        )?;

        final_key.enforce_equal(cs.ns(|| "DLOG hard part"), &combined_final_key)?;

        end_timer!(proof_time);

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