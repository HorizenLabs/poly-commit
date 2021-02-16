use algebra::{Field, AffineCurve, ProjectiveCurve, UniformRand};
use rand::{thread_rng, RngCore, SeedableRng};
use std::marker::PhantomData;
use poly_commit::{PCVerifierKey, PolynomialCommitment, LabeledCommitment, Evaluations, QuerySet};
use poly_commit::ipa_pc::{InnerProductArgPC, Commitment, Proof, VerifierKey, BatchProof};
use digest::Digest;
use criterion::*;
use rand_xorshift::XorShiftRng;
use blake2::Blake2s;

#[macro_use]
extern crate derivative;

#[derive(Clone, Default)]
struct BenchInfo {
    max_degree: usize,
    supported_degree: usize,
    num_commitments: usize,
    degree_bounds: Vec<bool>, // For each commitment, enforce or not a degree bound
    hiding_bound: bool,
    num_queries: usize,
}

#[derive(Derivative)]
#[derivative(Clone(bound = ""))]
struct BenchVerifierData<'a, F: Field, PC: PolynomialCommitment<F>> {
    vk:                      PC::VerifierKey,
    comms:                   Vec<LabeledCommitment<PC::Commitment>>,
    query_set:               QuerySet<'a, F>,
    values:                  Evaluations<'a, F>,
    proof:                   PC::BatchProof,
    opening_challenge:       F,
    _m:                      PhantomData<&'a F>, // To avoid compilation issue 'a
}

impl<'a, D, G> BenchVerifierData<'a, G::ScalarField, InnerProductArgPC<G, D>>
where
    G: AffineCurve,
    D: Digest,
{
    pub fn generate_vk<R: RngCore>(rng: &mut R, info: &BenchInfo) -> VerifierKey<G> {
        let BenchInfo {
            max_degree,
            supported_degree,
            ..
        } = info.clone();

        // Generate random params
        let pp = InnerProductArgPC::<G, D>::setup(max_degree, rng).unwrap();
        let (_, vk) = InnerProductArgPC::<G, D>::trim(
            &pp,
            supported_degree,
            0, // unused
            None, // unused
        ).unwrap();

        assert!(
            max_degree >= supported_degree,
            "max_degree < supported_degree"
        );
        
        vk
    }
    /// Generate dummy TestVerifierData, according to the specs defined in info.
    /// NOTE: The data is completely random, therefore the PC::batch_check_batch_proof
    /// verifier won't pass.
    pub fn get_random_data<R: RngCore>(rng: &mut R, info: &BenchInfo, vk: &VerifierKey<G>) -> Self {
        let BenchInfo {
            num_commitments,
            degree_bounds,
            hiding_bound,
            num_queries,
            ..
        } = info.clone();

        // Generate random labeled commitments
        assert_eq!(num_commitments, degree_bounds.len());

        let mut labeled_comms = Vec::new();
        let mut labels = Vec::new();

        for i in 0..num_commitments {
            let label = format!("Test_{}", i);
            labels.push(label.clone());

            let degree_bound_requested = degree_bounds[i];
            let comm = Commitment::<G>{
                comm: G::Projective::rand(rng).into_affine(),
                shifted_comm: if degree_bound_requested { Some(G::Projective::rand(rng).into_affine()) } else { None }
            };

            labeled_comms.push(LabeledCommitment::new(
                label,
                comm,
                if degree_bound_requested { Some(0) } else { None }
            ));
        }

        // Generate random query set and eval points
        let mut query_set = QuerySet::new();
        let mut values = Evaluations::new();

        for _ in 0..num_queries {
            let point = G::ScalarField::rand(rng);
            for (i, label) in labels.iter().enumerate() {
                query_set.insert((label.clone(), (format!("{}", i), point)));
                let value = G::ScalarField::rand(rng);
                values.insert((label.clone(), point), value);
            }
        }

        // Generate random proof
        let opening_challenge = G::ScalarField::rand(rng);
        let log_d = algebra::log2(vk.supported_degree() + 1) as usize;
        let proof = Proof::<G>{
            l_vec: vec![G::Projective::rand(rng).into_affine(); log_d],
            r_vec: vec![G::Projective::rand(rng).into_affine(); log_d],
            final_comm_key: G::Projective::rand(rng).into_affine(),
            c: G::ScalarField::rand(rng),
            hiding_comm: if hiding_bound { Some(G::Projective::rand(rng).into_affine()) } else { None },
            rand: if hiding_bound { Some(G::ScalarField::rand(rng)) } else { None },
        };
        let batch_proof = BatchProof::<G> {
            proof,
            batch_commitment: G::Projective::rand(rng).into_affine(),
            batch_values: vec![G::ScalarField::rand(rng); num_commitments],
        };

        Self {
            vk: vk.clone(),
            comms: labeled_comms,
            query_set,
            values,
            proof: batch_proof,
            opening_challenge,
            _m: PhantomData
        }
    }
}

fn bench_batch_verify_batch_proofs<G: AffineCurve, D: Digest>(
    c: &mut Criterion,
    bench_name: &str,
    max_degree: usize,
    num_commitments: usize,
    num_proofs_to_bench: Vec<usize>,
) {
    let rng = &mut XorShiftRng::seed_from_u64(1234567890u64);
    let mut group = c.benchmark_group(bench_name);

    let info = BenchInfo {
        max_degree,
        supported_degree: max_degree,
        num_commitments,
        degree_bounds: vec![false; num_commitments],
        hiding_bound: false, // Unless we use zk
        num_queries: 5
    };
    let vk = BenchVerifierData::<G::ScalarField, InnerProductArgPC<G, D>>::generate_vk(rng, &info);

    for num_proofs in num_proofs_to_bench.into_iter() {
        group.bench_with_input(BenchmarkId::from_parameter(num_proofs), &num_proofs, |bn, num_proofs| {
            bn.iter_batched(
                || {
                    let rng = &mut thread_rng();
                    let verifier_data_vec =
                        vec![
                            BenchVerifierData::<G::ScalarField, InnerProductArgPC<G, D>>::get_random_data(rng, &info, &vk);
                            *num_proofs
                        ];
                    let mut comms = Vec::new();
                    let mut query_sets = Vec::new();
                    let mut evals = Vec::new();
                    let mut proofs = Vec::new();
                    let mut opening_challenges = Vec::new();

                    verifier_data_vec.into_iter().for_each(|verifier_data| {
                        assert_eq!(&verifier_data.vk, &vk); // Vk should be equal for all proofs
                        comms.push(verifier_data.comms);
                        query_sets.push(verifier_data.query_set);
                        evals.push(verifier_data.values);
                        proofs.push(verifier_data.proof);
                        opening_challenges.push(verifier_data.opening_challenge.clone());
                    });

                    (comms, query_sets, evals, proofs, opening_challenges)
                },
                |(comms, query_sets, evals, proofs, opening_challenges)| {
                    let rng = &mut thread_rng();
                    InnerProductArgPC::<G, D>::batch_check_batch_proofs(
                        &vk,
                        comms.iter().map(|comm| comm.as_slice()).collect::<Vec<_>>(),
                        query_sets.iter(),
                        evals.iter(),
                        proofs.iter(),
                        opening_challenges,
                        rng
                    ).unwrap();
                },
                BatchSize::PerIteration
            );
        });
    }
    group.finish();
}

use algebra::curves::tweedle::{
        dee::Affine as TweedleDee,
        dum::Affine as TweedleDum,
    };

// the maximum degree we expect to handle is 2^19, maybe even below, e.g. 2^18
// Segment size |H| => 42, segment size |H|/2 => 84

fn bench_batch_verify_batch_proofs_tweedle_dee(c: &mut Criterion) {

    bench_batch_verify_batch_proofs::<TweedleDee, Blake2s>(
        c,
        "batch verification of batch proofs in tweedle-dee, |H| = segment_size = 1 << 19, number of proofs",
        1 << 19,
        42,
        vec![10, 50, 100],
    );

    bench_batch_verify_batch_proofs::<TweedleDee, Blake2s>(
        c,
        "batch verification of batch proofs in tweedle-dee, |H| = 1 << 19, segment_size = |H|/2, number of proofs",
        1 << 18,
        84,
        vec![10, 50, 100],
    );
}

fn bench_batch_verify_batch_proofs_tweedle_dum(c: &mut Criterion) {
    bench_batch_verify_batch_proofs::<TweedleDum, Blake2s>(
        c,
        "batch verification of batch proofs in tweedle-dum, |H| = segment_size = 1 << 19, number of proofs",
        1 << 19,
        42,
        vec![10, 50, 100],
    );

    bench_batch_verify_batch_proofs::<TweedleDum, Blake2s>(
        c,
        "batch verification of batch proofs in tweedle-dum, |H| = 1 << 19, segment_size = |H|/2, number of proofs",
        1 << 18,
        84,
        vec![10, 50, 100],
    );
}

criterion_group!(
name = tweedle_batch_verify_batch_proofs;
config = Criterion::default().sample_size(10);
targets = bench_batch_verify_batch_proofs_tweedle_dee, bench_batch_verify_batch_proofs_tweedle_dum
);

criterion_main!(tweedle_batch_verify_batch_proofs);