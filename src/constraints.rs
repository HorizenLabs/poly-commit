use crate::{
    data_structures::LabeledCommitment, BatchLCProof, LCTerm, LinearCombination,
    PolynomialCommitment
};
use algebra::{PrimeField, AffineCurve, Field};
use r1cs_std::{
    fields::{
        fp::FpGadget,
        nonnative::nonnative_field_gadget::NonNativeFieldGadget
    },
    prelude::*,
};
use r1cs_core::{ConstraintSystem, SynthesisError};
use std::{
    collections::{HashMap, HashSet},
    borrow::Borrow, cmp::Eq, cmp::PartialEq, hash::Hash, marker::Sized
};
use crate::fiat_shamir::constraints::FiatShamirRngGadget;

/// Define the minimal interface of prepared allocated structures.
pub trait PrepareGadget<Unprepared, ConstraintF: PrimeField>: Sized {
    /// Prepare from an unprepared element.
    fn prepare<CS: ConstraintSystem<ConstraintF>>(
        cs: CS,
        unprepared: &Unprepared
    ) -> Result<Self, SynthesisError>;
}

/// A coefficient of `LinearCombination`.
#[derive(Clone)]
pub enum LinearCombinationCoeffGadget<SimulationF: PrimeField, ConstraintF: PrimeField> {
    /// Coefficient 1.
    One,
    /// Coefficient -1.
    MinusOne,
    /// Other coefficient, represented as a nonnative field element.
    Gadget(NonNativeFieldGadget<SimulationF, ConstraintF>),
}

/// An allocated version of `LinearCombination`.
#[derive(Clone)]
pub struct LinearCombinationGadget<SimulationF: PrimeField, ConstraintF: PrimeField> {
    /// The label.
    pub label: String,
    /// The linear combination of `(coeff, poly_label)` pairs.
    pub terms: Vec<(LinearCombinationCoeffGadget<SimulationF, ConstraintF>, LCTerm)>,
}

impl<SimulationF: PrimeField, ConstraintF: PrimeField> AllocGadget<LinearCombination<SimulationF>, ConstraintF>
for LinearCombinationGadget<SimulationF, ConstraintF>
{
    fn alloc<F, T, CS: ConstraintSystem<ConstraintF>>(mut cs: CS, f: F) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<LinearCombination<SimulationF>>
    {
        let LinearCombination { label, terms } = f()?.borrow().clone();

        let new_terms: Vec<(LinearCombinationCoeffGadget<SimulationF, ConstraintF>, LCTerm)> = terms
            .iter()
            .enumerate()
            .map(|(i, term)| {
                let (f, lc_term) = term;

                let fg = NonNativeFieldGadget::alloc(
                    cs.ns(|| format!("alloc LC term {}", i)),
                    || Ok(f)
                ).unwrap();

                (LinearCombinationCoeffGadget::Gadget(fg), lc_term.clone())
            })
            .collect();

        Ok(Self {
            label,
            terms: new_terms,
        })
    }

    fn alloc_input<F, T, CS: ConstraintSystem<ConstraintF>>(mut cs: CS, f: F) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<LinearCombination<SimulationF>>
    {
        let LinearCombination { label, terms } = f()?.borrow().clone();

        let new_terms: Vec<(LinearCombinationCoeffGadget<SimulationF, ConstraintF>, LCTerm)> = terms
            .iter()
            .enumerate()
            .map(|(i, term)| {
                let (f, lc_term) = term;

                let fg = NonNativeFieldGadget::alloc_input(
                    cs.ns(|| format!("alloc LC term {}", i)),
                    || Ok(f)
                ).unwrap();

                (LinearCombinationCoeffGadget::Gadget(fg), lc_term.clone())
            })
            .collect();

        Ok(Self {
            label,
            terms: new_terms,
        })
    }
}

/// Describes the interface for a gadget for a `PolynomialCommitment`
/// verifier.
pub trait PolynomialCommitmentGadget<
    G:  AffineCurve,
    PC: PolynomialCommitment<G>,
>: Clone
{
    /// An allocated version of `PC::VerifierKey`.
    type VerifierKeyGadget: AllocGadget<PC::VerifierKey, <G::BaseField as Field>::BasePrimeField>
                            + Clone
                            + ToBytesGadget<<G::BaseField as Field>::BasePrimeField>;
    
    /// An allocated version of `PC::PreparedVerifierKey`.
    type PreparedVerifierKeyGadget: AllocGadget<PC::PreparedVerifierKey, <G::BaseField as Field>::BasePrimeField>
                                    + Clone
                                    + PrepareGadget<Self::VerifierKeyGadget, <G::BaseField as Field>::BasePrimeField>;

    /// An allocated version of `PC::Commitment`.
    type CommitmentGadget: AllocGadget<PC::Commitment, <G::BaseField as Field>::BasePrimeField>
                           + Clone
                           + ToBytesGadget<<G::BaseField as Field>::BasePrimeField>;

    /// An allocated version of `PC::PreparedCommitment`.
    type PreparedCommitmentGadget: AllocGadget<PC::PreparedCommitment, <G::BaseField as Field>::BasePrimeField>
                                   + PrepareGadget<Self::CommitmentGadget, <G::BaseField as Field>::BasePrimeField>
                                   + Clone;

    /// An allocated version of `LabeledCommitment<PC::Commitment>`.
    type LabeledCommitmentGadget: AllocGadget<LabeledCommitment<PC::Commitment>, <G::BaseField as Field>::BasePrimeField>
                                  + Clone;

    /// A prepared, allocated version of `LabeledCommitment<PC::Commitment>`.
    type PreparedLabeledCommitmentGadget: Clone;

    /// A FiatShamirRngGadget, providing a source of random data used in the polynomial commitment checking.
    type RandomOracleGadget: FiatShamirRngGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>;

    /// An allocated version of `PC::Proof`.
    type ProofGadget: AllocGadget<PC::Proof, <G::BaseField as Field>::BasePrimeField> + Clone;

    /// An allocated version of `PC::BatchLCProof`.
    type BatchLCProofGadget: AllocGadget<BatchLCProof<G, PC>, <G::BaseField as Field>::BasePrimeField>
                             + Clone;

    /// Add to `CS` new constraints that check that `proof_i` is a valid evaluation proof
    /// at `point_i` for the polynomial in `commitment_i`.
    fn batch_check_evaluations<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        cs:                 CS,
        verification_key:   &Self::VerifierKeyGadget,
        commitments:        &[Self::LabeledCommitmentGadget],
        query_set:          &QuerySetGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,
        evaluations:        &EvaluationsGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,
        proofs:             &[Self::ProofGadget],
        fs_rng:             &mut Self::RandomOracleGadget,
    ) -> Result<Boolean, SynthesisError>;

    /// Add to `CS` new constraints that conditionally check that `proof` is a valid evaluation
    /// proof at the points in `query_set` for the combinations `linear_combinations`.
    fn prepared_check_combinations<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        cs: CS,
        prepared_verification_key:  &Self::PreparedVerifierKeyGadget,
        linear_combinations:        &[LinearCombinationGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>],
        prepared_commitments:       &[Self::PreparedLabeledCommitmentGadget],
        query_set:                  &QuerySetGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,
        evaluations:                &EvaluationsGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,
        proof:                      &Self::BatchLCProofGadget,
        fs_rng:             &mut Self::RandomOracleGadget,
    ) -> Result<Boolean, SynthesisError>;

    /// Create the labeled commitment gadget from the commitment gadget
    fn create_labeled_commitment(
        label: String,
        commitment: Self::CommitmentGadget,
        degree_bound: Option<FpGadget<<G::BaseField as Field>::BasePrimeField>>,
    ) -> Self::LabeledCommitmentGadget;

    /// Create the prepared labeled commitment gadget from the commitment gadget
    fn create_prepared_labeled_commitment(
        label: String,
        commitment: Self::PreparedCommitmentGadget,
        degree_bound: Option<FpGadget<<G::BaseField as Field>::BasePrimeField>>,
    ) -> Self::PreparedLabeledCommitmentGadget;
}

#[derive(Clone, Hash, PartialEq, Eq)]
/// A labeled point variable, for queries to a polynomial commitment.
pub struct LabeledPointGadget<SimulationF: PrimeField, ConstraintF: PrimeField> {
    /// The label of the point.
    /// MUST be a unique identifier in a query set.
    pub name: String,
    /// The point value.
    pub value: NonNativeFieldGadget<SimulationF, ConstraintF>,
}

/// An allocated version of `QuerySet`.
#[derive(Clone)]
pub struct QuerySetGadget<SimulationF: PrimeField, ConstraintF: PrimeField>(
    pub HashSet<(String, LabeledPointGadget<SimulationF, ConstraintF>)>
);

/// An allocated version of `Evaluations`.
#[derive(Clone)]
pub struct EvaluationsGadget<SimulationF: PrimeField, ConstraintF: PrimeField>(
    pub HashMap<LabeledPointGadget<SimulationF, ConstraintF>, NonNativeFieldGadget<SimulationF, ConstraintF>>
);

impl<SimulationF: PrimeField, ConstraintF: PrimeField> EvaluationsGadget<SimulationF, ConstraintF> {
    /// find the evaluation result
    pub fn get_lc_eval(
        &self,
        lc_string: &str,
        point: &NonNativeFieldGadget<SimulationF, ConstraintF>,
    ) -> Result<NonNativeFieldGadget<SimulationF, ConstraintF>, SynthesisError> {
        let key = LabeledPointGadget::<SimulationF, ConstraintF> {
            name: String::from(lc_string),
            value: point.clone(),
        };
        Ok(self.0.get(&key).map(|v| (*v).clone()).unwrap())
    }
}
