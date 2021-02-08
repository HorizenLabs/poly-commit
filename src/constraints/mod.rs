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
    type PreparedLabeledCommitmentGadget: PrepareGadget<Self::LabeledCommitmentGadget, <G::BaseField as Field>::BasePrimeField>
                                          + Clone;

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
    ) -> Result<(), SynthesisError>;

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
    ) -> Result<(), SynthesisError>;

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


#[cfg(test)]
pub mod test {
    use crate::*;
    use algebra::{Field, UniformRand};
    use rand::{distributions::Distribution, thread_rng};
    use crate::{
        fiat_shamir::constraints::FiatShamirRngGadget,
        constraints::{
            PrepareGadget, PolynomialCommitmentGadget, QuerySetGadget,
            EvaluationsGadget, LabeledPointGadget
        }
    };
    use r1cs_std::{
        alloc::AllocGadget,
        fields::nonnative::nonnative_field_gadget::NonNativeFieldGadget,
        test_constraint_system::TestConstraintSystem
    };
    use r1cs_core::ConstraintSystem;
    use std::collections::{HashSet, HashMap};

    #[derive(Default)]
    pub struct TestInfo {
        num_iters: usize,
        max_degree: Option<usize>,
        supported_degree: Option<usize>,
        num_polynomials: usize,
        enforce_degree_bounds: bool,
        max_num_queries: usize,
    }

    pub fn bad_degree_bound_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest
    {
        let rng = &mut thread_rng();
        let max_degree = 100;
        let pp = PC::setup::<_, D>(max_degree, rng)?;

        for _ in 0..10 {
            let mut cs = TestConstraintSystem::<<G::BaseField as Field>::BasePrimeField>::new();

            let supported_degree = rand::distributions::Uniform::from(1..=max_degree).sample(rng);
            assert!(
                max_degree >= supported_degree,
                "max_degree < supported_degree"
            );

            let mut labels = Vec::new();
            let mut polynomials = Vec::new();
            let mut degree_bounds = Vec::new();

            for i in 0..10 {
                let label = format!("Test{}", i);
                labels.push(label.clone());
                let poly = Polynomial::rand(supported_degree, rng);

                let degree_bound = 1usize;
                let hiding_bound = Some(1);
                degree_bounds.push(degree_bound);

                polynomials.push(LabeledPolynomial::new(
                    label,
                    poly,
                    Some(degree_bound),
                    hiding_bound,
                ))
            }

            let supported_hiding_bound = polynomials
                .iter()
                .map(|p| p.hiding_bound().unwrap_or(0))
                .max()
                .unwrap_or(0);
            println!("supported degree: {:?}", supported_degree);
            println!("supported hiding bound: {:?}", supported_hiding_bound);
            let (ck, vk) = PC::trim(
                &pp,
                supported_degree,
                supported_hiding_bound,
                Some(degree_bounds.as_slice()),
            )?;
            println!("Trimmed");

            let vk_gadget = PCG::VerifierKeyGadget::alloc(
                cs.ns(|| "alloc vk"),
                || Ok(vk)
            ).unwrap();

            let prepared_vk_gadget = PCG::PreparedVerifierKeyGadget::prepare(
                cs.ns(|| "prepare vk"),
                &vk_gadget
            ).unwrap();

            let (comms, rands) = PC::commit(&ck, &polynomials, Some(rng))?;

            let comms_gs = comms.iter().enumerate().map(|(i, comm)| {
                let comm = PCG::LabeledCommitmentGadget::alloc(
                    cs.ns(|| format!("alloc comm {}", i)),
                    || Ok(comm.clone())
                ).unwrap();

                PCG::PreparedLabeledCommitmentGadget::prepare(
                    cs.ns(|| format!("prepare comm {}", i)),
                    &comm
                ).unwrap()
            }).collect::<Vec<_>>();

            let mut query_set = QuerySet::new();
            let mut query_set_gadget = QuerySetGadget::<G::ScalarField, <G::BaseField as Field>::BasePrimeField>{ 0: HashSet::new() };

            let mut values = Evaluations::new();
            let mut evaluations_gadget = EvaluationsGadget::<G::ScalarField, <G::BaseField as Field>::BasePrimeField>{ 0: HashMap::new() };

            let point = G::ScalarField::rand(rng);
            let point_gadget = NonNativeFieldGadget::<G::ScalarField, <G::BaseField as Field>::BasePrimeField>::alloc(
                cs.ns(|| "alloc challenge point"),
                || Ok(point)
            ).unwrap();

            for (i, label) in labels.iter().enumerate() {
                let point_label = format!("{}", i);

                query_set.insert((label.clone(), (point_label.clone(), point)));
                query_set_gadget.0.insert((label.clone(), LabeledPointGadget(point_label, point_gadget.clone())));

                let value = polynomials[i].evaluate(point);
                let value_gadget = NonNativeFieldGadget::<G::ScalarField, <G::BaseField as Field>::BasePrimeField>::alloc(
                    cs.ns(|| format!("alloc eval_{}", i)),
                    || Ok(value)
                ).unwrap();

                values.insert((label.clone(), point), value);
                evaluations_gadget.0.insert(LabeledPointGadget(label.to_string(), point_gadget.clone()), value_gadget);
            }
            println!("Generated query set");

            let fs_rng = &mut PC::RandomOracle::new();
            let proof = PC::batch_open_individual_opening_challenges(
                &ck,
                &polynomials,
                &comms,
                &query_set,
                &rands,
                Some(rng),
                fs_rng
            )?;

            let proof_gadget = PCG::BatchProofGadget::alloc(
                cs.ns(|| "alloc proof"),
                || Ok(proof)
            ).unwrap();

            let fs_rng_gadget = &mut PCG::RandomOracleGadget::new(
                cs.ns(|| "alloc fs_rng")
            ).unwrap();

            PCG::prepared_batch_check_individual_opening_challenges(
                cs.ns(|| "batch verify"),
                &prepared_vk_gadget,
                comms_gs.as_slice(),
                &query_set_gadget,
                &evaluations_gadget,
                &proof_gadget,
                fs_rng_gadget,
            ).unwrap();

            if !cs.is_satisfied() {
                println!("{:?}", cs.which_is_unsatisfied());
            }

            assert!(cs.is_satisfied(), "proof was incorrect, Query set: {:#?}", query_set);
        }
        Ok(())
    }

    pub fn test_template<G, PC, PCG, D>(info: TestInfo) -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let TestInfo {
            num_iters,
            max_degree,
            supported_degree,
            num_polynomials,
            enforce_degree_bounds,
            max_num_queries,
            ..
        } = info;

        let rng = &mut thread_rng();

        let max_degree =
            max_degree.unwrap_or(rand::distributions::Uniform::from(2..=64).sample(rng));
        let pp = PC::setup::<_, D>(max_degree, rng)?;

        for _ in 0..num_iters {
            let mut cs = TestConstraintSystem::<<G::BaseField as Field>::BasePrimeField>::new();

            let supported_degree = match supported_degree {
                Some(0) => 0,
                Some(d) => d,
                None => rand::distributions::Uniform::from(1..=max_degree).sample(rng)
            };
            assert!(
                max_degree >= supported_degree,
                "max_degree < supported_degree"
            );
            let mut polynomials = Vec::new();
            let mut degree_bounds = if enforce_degree_bounds {
                Some(Vec::new())
            } else {
                None
            };

            let mut labels = Vec::new();
            println!("Sampled supported degree");

            // Generate polynomials
            let num_points_in_query_set =
                rand::distributions::Uniform::from(1..=max_num_queries).sample(rng);
            for i in 0..num_polynomials {
                let label = format!("Test{}", i);
                labels.push(label.clone());
                let degree = if supported_degree > 0 {
                    rand::distributions::Uniform::from(1..=supported_degree).sample(rng)
                } else {
                    0
                };
                let poly = Polynomial::rand(degree, rng);

                let degree_bound = if let Some(degree_bounds) = &mut degree_bounds {
                    let range = rand::distributions::Uniform::from(degree..=supported_degree);
                    let degree_bound = range.sample(rng);
                    degree_bounds.push(degree_bound);
                    Some(degree_bound)
                } else {
                    None
                };

                let hiding_bound = if num_points_in_query_set >= degree {
                    Some(degree)
                } else {
                    Some(num_points_in_query_set)
                };
                println!("Hiding bound: {:?}", hiding_bound);

                polynomials.push(LabeledPolynomial::new(
                    label,
                    poly,
                    degree_bound,
                    hiding_bound,
                ))
            }
            let supported_hiding_bound = polynomials
                .iter()
                .map(|p| p.hiding_bound().unwrap_or(0))
                .max()
                .unwrap_or(0);
            println!("supported degree: {:?}", supported_degree);
            println!("supported hiding bound: {:?}", supported_hiding_bound);
            println!("num_points_in_query_set: {:?}", num_points_in_query_set);
            let (ck, vk) = PC::trim(
                &pp,
                supported_degree,
                supported_hiding_bound,
                degree_bounds.as_ref().map(|s| s.as_slice()),
            )?;
            println!("Trimmed");

            let vk_gadget = PCG::VerifierKeyGadget::alloc(
                cs.ns(|| "alloc vk"),
                || Ok(vk)
            ).unwrap();

            let prepared_vk_gadget = PCG::PreparedVerifierKeyGadget::prepare(
                cs.ns(|| "prepare vk"),
                &vk_gadget
            ).unwrap();

            let (comms, rands) = PC::commit(&ck, &polynomials, Some(rng))?;

            let comms_gs = comms.iter().enumerate().map(|(i, comm)| {
                let comm = PCG::LabeledCommitmentGadget::alloc(
                    cs.ns(|| format!("alloc comm {}", i)),
                    || Ok(comm.clone())
                ).unwrap();

                PCG::PreparedLabeledCommitmentGadget::prepare(
                    cs.ns(|| format!("prepare comm {}", i)),
                    &comm
                ).unwrap()
            }).collect::<Vec<_>>();

            let mut query_set = QuerySet::new();
            let mut query_set_gadget = QuerySetGadget::<G::ScalarField, <G::BaseField as Field>::BasePrimeField>{ 0: HashSet::new() };

            let mut values = Evaluations::new();
            let mut evaluations_gadget = EvaluationsGadget::<G::ScalarField, <G::BaseField as Field>::BasePrimeField>{ 0: HashMap::new() };

            let point = G::ScalarField::rand(rng);
            let point_gadget = NonNativeFieldGadget::<G::ScalarField, <G::BaseField as Field>::BasePrimeField>::alloc(
                cs.ns(|| "alloc challenge point"),
                || Ok(point)
            ).unwrap();

            for (i, label) in labels.iter().enumerate() {
                let point_label = format!("{}", i);

                query_set.insert((label.clone(), (point_label.clone(), point)));
                query_set_gadget.0.insert((label.clone(), LabeledPointGadget(point_label, point_gadget.clone())));

                let value = polynomials[i].evaluate(point);
                let value_gadget = NonNativeFieldGadget::<G::ScalarField, <G::BaseField as Field>::BasePrimeField>::alloc(
                    cs.ns(|| format!("alloc eval_{}", i)),
                    || Ok(value)
                ).unwrap();

                values.insert((label.clone(), point), value);
                evaluations_gadget.0.insert(LabeledPointGadget(label.to_string(), point_gadget.clone()), value_gadget);
            }
            println!("Generated query set");

            let fs_rng = &mut PC::RandomOracle::new();
            let proof = PC::batch_open(
                &ck,
                &polynomials,
                &comms,
                &query_set,
                &rands,
                Some(rng),
                fs_rng
            )?;

            let proof_gadget = PCG::BatchProofGadget::alloc(
                cs.ns(|| "alloc proof"),
                || Ok(proof)
            ).unwrap();

            let fs_rng_gadget = &mut PCG::RandomOracleGadget::new(
                cs.ns(|| "alloc fs_rng")
            ).unwrap();

            PCG::prepared_batch_check_individual_opening_challenges(
                cs.ns(|| "batch verify"),
                &prepared_vk_gadget,
                comms_gs.as_slice(),
                &query_set_gadget,
                &evaluations_gadget,
                &proof_gadget,
                fs_rng_gadget,
            ).unwrap();
            
            if !cs.is_satisfied() {
                println!("{:?}", cs.which_is_unsatisfied());
                println!(
                    "Failed with {} polynomials, num_points_in_query_set: {:?}",
                    num_polynomials, num_points_in_query_set
                );
                println!("Degree of polynomials:",);
                for poly in polynomials {
                    println!("Degree: {:?}", poly.degree());
                }
            }
            assert!(cs.is_satisfied(), "proof was incorrect, Query set: {:#?}", query_set);
        }
        Ok(())
    }

    pub fn constant_poly_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let info = TestInfo {
            num_iters: 1,
            max_degree: None,
            supported_degree: Some(0),
            num_polynomials: 1,
            enforce_degree_bounds: false,
            max_num_queries: 1,
            ..Default::default()
        };
        test_template::<G, PC, PCG, D>(info)
    }

    pub fn single_poly_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let info = TestInfo {
            num_iters: 1,
            max_degree: None,
            supported_degree: None,
            num_polynomials: 1,
            enforce_degree_bounds: false,
            max_num_queries: 1,
            ..Default::default()
        };
        test_template::<G, PC, PCG, D>(info)
    }

    pub fn linear_poly_degree_bound_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let info = TestInfo {
            num_iters: 1,
            max_degree: Some(2),
            supported_degree: Some(1),
            num_polynomials: 1,
            enforce_degree_bounds: true,
            max_num_queries: 1,
            ..Default::default()
        };
        test_template::<G, PC, PCG, D>(info)
    }

    pub fn single_poly_degree_bound_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let info = TestInfo {
            num_iters: 1,
            max_degree: None,
            supported_degree: None,
            num_polynomials: 1,
            enforce_degree_bounds: true,
            max_num_queries: 1,
            ..Default::default()
        };
        test_template::<G, PC, PCG, D>(info)
    }

    pub fn quadratic_poly_degree_bound_multiple_queries_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let info = TestInfo {
            num_iters: 1,
            max_degree: Some(3),
            supported_degree: Some(2),
            num_polynomials: 1,
            enforce_degree_bounds: true,
            max_num_queries: 2,
            ..Default::default()
        };
        test_template::<G, PC, PCG, D>(info)
    }

    pub fn single_poly_degree_bound_multiple_queries_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let info = TestInfo {
            num_iters: 1,
            max_degree: None,
            supported_degree: None,
            num_polynomials: 1,
            enforce_degree_bounds: true,
            max_num_queries: 2,
            ..Default::default()
        };
        test_template::<G, PC, PCG, D>(info)
    }

    pub fn two_polys_degree_bound_single_query_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let info = TestInfo {
            num_iters: 1,
            max_degree: None,
            supported_degree: None,
            num_polynomials: 2,
            enforce_degree_bounds: true,
            max_num_queries: 1,
            ..Default::default()
        };
        test_template::<G, PC, PCG, D>(info)
    }

    pub fn full_end_to_end_test<G, PC, PCG, D>() -> Result<(), PC::Error>
        where
            G: AffineCurve,
            PC: PolynomialCommitment<G>,
            PCG: PolynomialCommitmentGadget<G, PC>,
            D: Digest,
    {
        let info = TestInfo {
            num_iters: 1,
            max_degree: None,
            supported_degree: None,
            num_polynomials: 10,
            enforce_degree_bounds: true,
            max_num_queries: 5,
            ..Default::default()
        };
        test_template::<G, PC, PCG, D>(info)
    }
}