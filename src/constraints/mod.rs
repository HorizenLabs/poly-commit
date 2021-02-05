use crate::{PolynomialCommitment, LabeledCommitment};
use algebra::{AffineCurve, Field};
use r1cs_std::{
    fields::{
        fp::FpGadget,
        nonnative::nonnative_field_gadget::NonNativeFieldGadget
    },
    prelude::*,
};
use r1cs_core::{ConstraintSystem, SynthesisError};
use std::marker::Sized;
use crate::fiat_shamir::constraints::FiatShamirRngGadget;

mod data_structures;
pub use self::data_structures::*;

/// Describes the interface for a gadget for a `PolynomialCommitment`
/// verifier.
pub trait PolynomialCommitmentGadget<
    G:  AffineCurve,
    PC: PolynomialCommitment<G>,
>: Sized
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

    /// The evaluation proof for a query set.
    type BatchProofGadget: AllocGadget<PC::BatchProof, <G::BaseField as Field>::BasePrimeField> + Clone;

    /// Add to `CS` new constraints that check that `proof` is a valid evaluation proof
    /// at points in `query_set` for the polynomials in `commitments`.
    fn prepared_batch_check_individual_opening_challenges<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        cs:                 CS,
        verification_key:   &Self::PreparedVerifierKeyGadget,
        commitments:        &[Self::PreparedLabeledCommitmentGadget],
        query_set:          &QuerySetGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,
        evaluations:        &EvaluationsGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,
        proof:              &Self::BatchProofGadget,
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
        proof:                      &BatchLCProofGadget<G, PC, Self>,
        fs_rng:                     &mut Self::RandomOracleGadget,
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

    /// Given the coordinates of a polynomial with respect to the Lagrange Basis
    /// over a given FFT domain, enforce that the polynomial commits to an expected
    /// value, using only the commitments of the Lagrange Basis.
    /// This is an optimization coming from MinaProtocol: naive procedure would
    /// require to enforce the interpolation of the polynomial from its evaluations,
    /// and then enforce the commitment of its coefficients.
    fn verify_polynomial_commitment_from_lagrange_representation<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        cs:                  CS,
        expected_comm:       &Self::CommitmentGadget,
        lagrange_poly_comms: &[PC::Commitment],
        poly_coords:         &[NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>],
    ) -> Result<(), SynthesisError>;
}