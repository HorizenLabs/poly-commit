use algebra::{AffineCurve, UniformRand};
use algebra_utils::DensePolynomial as Polynomial;
use rand::{thread_rng, RngCore, SeedableRng};
use poly_commit::{PolynomialCommitment, LabeledPolynomial};
use poly_commit::ipa_pc::{InnerProductArgPC, CommitterKey};
use digest::Digest;
use criterion::*;
use rand_xorshift::XorShiftRng;
use blake2::Blake2s;

#[derive(Clone, Default)]
struct BenchInfo {
    max_degree: usize,
    supported_degree: usize
}

fn generate_ck<G: AffineCurve, D: Digest, R: RngCore>(rng: &mut R, info: &BenchInfo) -> CommitterKey<G> {
    let BenchInfo {
        max_degree,
        supported_degree,
        ..
    } = info.clone();

    // Generate random params
    let pp = InnerProductArgPC::<G, D>::setup(max_degree, rng).unwrap();
    let (ck, _) = InnerProductArgPC::<G, D>::trim(
        &pp,
        supported_degree,
        0, // unused
        None, // unused
    ).unwrap();

    assert!(
        max_degree >= supported_degree,
        "max_degree < supported_degree"
    );
    
    ck
}

fn bench_open_proof<G: AffineCurve, D: Digest>(
    c: &mut Criterion,
    bench_name: &str,
    coeffs: usize,
) {
    let rng = &mut XorShiftRng::seed_from_u64(1234567890u64);
    let mut group = c.benchmark_group(bench_name);

    let max_degree = coeffs - 1;

    let info = BenchInfo {
        max_degree,
        supported_degree: max_degree
    };
    let ck = generate_ck::<G, D, XorShiftRng>(rng, &info);

    group.bench_with_input(BenchmarkId::from_parameter(max_degree), &max_degree, |bn, max_degree| {
        bn.iter_batched(
            || {

                let rng = &mut thread_rng();
                let mut polynomials = Vec::new();

                let label = format!("Test");

                polynomials.push(LabeledPolynomial::new(
                    label,
                    Polynomial::<G::ScalarField>::rand(*max_degree, rng),
                    None,
                    None,
                ));
        
                let (comms, rands) = InnerProductArgPC::<G, D>::commit(&ck, &polynomials, Some(rng)).unwrap();

                let point = G::ScalarField::rand(rng);
                let opening_challenge = G::ScalarField::rand(rng);

                (polynomials, comms, point, opening_challenge, rands)
            },
            |(polynomials, comms, point, opening_challenge, rands)| {
                let rng = &mut thread_rng();
                InnerProductArgPC::<G, D>::open(
                    &ck,
                    &polynomials,
                    &comms,
                    point,
                    opening_challenge,
                    &rands,
                    Some(rng)
                ).unwrap();
            },
            BatchSize::PerIteration
        );
    });
    group.finish();
}

use algebra::curves::tweedle::{
        dee::Affine as TweedleDee,
        dum::Affine as TweedleDum,
    };

fn bench_open_proof_tweedle_dee(c: &mut Criterion) {

    for n in 16..22 {
        bench_open_proof::<TweedleDee, Blake2s>(
            c,
            "open proof in tweedle-dee, coeffs",
            1 << n
        );
    }
}

fn bench_open_proof_tweedle_dum(c: &mut Criterion) {

    for n in 16..22 {
        bench_open_proof::<TweedleDum, Blake2s>(
            c,
            "open proof in tweedle-dum, coeffs",
            1 << n
        );    
    }
}

criterion_group!(
name = tweedle_open_proof;
config = Criterion::default().sample_size(10);
targets = bench_open_proof_tweedle_dee, bench_open_proof_tweedle_dum
);

criterion_main!(tweedle_open_proof);