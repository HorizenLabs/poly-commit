mod data_structures;
pub use self::data_structures::*;
use algebra::{PrimeField, AffineCurve};
use r1cs_std::groups::GroupGadget;
use crate::fiat_shamir::constraints::FiatShamirRngGadget;
use r1cs_core::{ToConstraintField, ConstraintSystem, SynthesisError};
use crate::fiat_shamir::FiatShamirRng;
use crate::constraints::{PolynomialCommitmentGadget, QuerySetGadget, EvaluationsGadget, LinearCombinationGadget, BatchLCProofGadget};
use crate::ipa_pc::{InnerProductArgPC, Commitment};
use r1cs_std::to_field_gadget_vec::ToConstraintFieldGadget;
use r1cs_std::bits::boolean::Boolean;
use r1cs_std::fields::fp::FpGadget;
use r1cs_std::fields::nonnative::nonnative_field_gadget::NonNativeFieldGadget;
use std::marker::PhantomData;

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
        _cs: CS,
        _verification_key: &Self::PreparedVerifierKeyGadget,
        _commitments:      &[Self::PreparedLabeledCommitmentGadget],
        _query_set:        &QuerySetGadget<F, ConstraintF>,
        _evaluations:      &EvaluationsGadget<F, ConstraintF>,
        _proof:            &Self::BatchProofGadget,
        _fs_rng:           &mut Self::RandomOracleGadget
    ) -> Result<Boolean, SynthesisError>
    {
        unimplemented!()
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
    ) -> Result<Boolean, SynthesisError>
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
        use algebra::{
            UniformRand, ProjectiveCurve
        };
        use r1cs_std::ToBitsGadget;

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
        result.sub(cs.ns(|| format!("subtract shift")), &shift)?;

        result.enforce_equal(cs.ns(|| "actual_comm == expected_comm"), &expected_comm.comm)?;

        Ok(())
    }
}