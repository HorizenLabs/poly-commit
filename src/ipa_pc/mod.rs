use crate::{BTreeMap, BTreeSet, String, ToString, Vec, PublicAccumulationScheme};
use crate::{BatchLCProof, Error, Evaluations, QuerySet};
use crate::{LabeledCommitment, LabeledPolynomial, LinearCombination};
use crate::{PCCommitterKey, PCRandomness, PCUniversalParams, Polynomial, PolynomialCommitment};

use algebra_utils::msm::VariableBaseMSM;
use algebra::{ToBytes, to_bytes, Field, PrimeField, UniformRand, Group, AffineCurve, ProjectiveCurve};
use std::{format, vec};
use std::{convert::TryInto, marker::PhantomData};
use rand_core::RngCore;

mod data_structures;
pub use data_structures::*;

use rayon::prelude::*;

#[cfg(feature = "gpu")]
use algebra_kernels::polycommit::{get_kernels, get_gpu_min_length};

use digest::Digest;
use rand_core::OsRng;

/// A polynomial commitment scheme based on the hardness of the
/// discrete logarithm problem in prime-order groups.
/// The construction is described in detail in [[BCMS20]][pcdas].
///
/// Degree bound enforcement requires that (at least one of) the points at
/// which a committed polynomial is evaluated are from a distribution that is
/// random conditioned on the polynomial. This is because degree bound
/// enforcement relies on checking a polynomial identity at this point.
/// More formally, the points must be sampled from an admissible query sampler,
/// as detailed in [[CHMMVW20]][marlin].
///
/// [pcdas]: https://eprint.iacr.org/2020/499
/// [marlin]: https://eprint.iacr.org/2019/1047
pub struct InnerProductArgPC<G: AffineCurve, D: Digest> {
    _projective: PhantomData<G>,
    _digest: PhantomData<D>,
}

impl<G: AffineCurve, D: Digest> InnerProductArgPC<G, D> {
    /// `PROTOCOL_NAME` is used as a seed for the setup function.
    pub const PROTOCOL_NAME: &'static [u8] = b"PC-DL-2020";

    /// Create a Pedersen commitment to `scalars` using the commitment key `comm_key`.
    /// Optionally, randomize the commitment using `hiding_generator` and `randomizer`.
    fn cm_commit(
        comm_key: &[G],
        scalars: &[G::ScalarField],
        hiding_generator: Option<G>,
        randomizer: Option<G::ScalarField>,
    ) -> G::Projective {
        let scalars_bigint = scalars.par_iter()
            .map(|s| s.into_repr())
            .collect::<Vec<_>>();
        let mut comm = VariableBaseMSM::multi_scalar_mul(comm_key, &scalars_bigint);
        if randomizer.is_some() {
            assert!(hiding_generator.is_some());
            comm += &hiding_generator.unwrap().mul(randomizer.unwrap());
        }

        comm
    }

    fn compute_random_oracle_challenge(bytes: &[u8]) -> G::ScalarField {
        let mut i = 0u64;
        let mut challenge = None;
        while challenge.is_none() {
            let hash_input = to_bytes![bytes, i].unwrap();
            let hash = D::digest(&hash_input);
            challenge = <G::ScalarField as PrimeField>::from_random_bytes(&hash);

            i += 1;
        }

        challenge.unwrap()
    }

    #[inline]
    fn inner_product(l: &[G::ScalarField], r: &[G::ScalarField]) -> G::ScalarField {
        l.par_iter().zip(r).map(|(li, ri)| *li * ri).sum()
    }

    /// The succinct portion of `PC::check`. This algorithm runs in time
    /// O(log d), where d is the degree of the committed polynomials.
    fn succinct_check<'a>(
        vk: &VerifierKey<G>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Commitment<G>>>,
        point: G::ScalarField,
        values: impl IntoIterator<Item = G::ScalarField>,
        proof: &Proof<G>,
        opening_challenges: &dyn Fn(u64) -> G::ScalarField,
    ) -> Option<SuccinctCheckPolynomial<G::ScalarField>> {
        let check_time = start_timer!(|| "Succinct checking");

        let d = vk.supported_degree();

        // `log_d` is ceil(log2 (d + 1)), which is the number of steps to compute all of the challenges
        let log_d = algebra::log2(d + 1) as usize;

        let mut combined_commitment_proj = <G::Projective as ProjectiveCurve>::zero();
        let mut combined_v = G::ScalarField::zero();

        let mut opening_challenge_counter = 0;
        let mut cur_challenge = opening_challenges(opening_challenge_counter);
        opening_challenge_counter += 1;

        let labeled_commitments = commitments.into_iter();
        let values = values.into_iter();

        for (labeled_commitment, value) in labeled_commitments.zip(values) {
            let commitment = labeled_commitment.commitment();
            combined_v += &(cur_challenge * &value);
            combined_commitment_proj += &labeled_commitment.commitment().comm.mul(cur_challenge);
            cur_challenge = opening_challenges(opening_challenge_counter);
            opening_challenge_counter += 1;

            let degree_bound = labeled_commitment.degree_bound();
            assert_eq!(degree_bound.is_some(), commitment.shifted_comm.is_some());

            if let Some(degree_bound) = degree_bound {
                let shift = point.pow([(vk.supported_degree() - degree_bound) as u64]);
                combined_v += &(cur_challenge * &value * &shift);
                combined_commitment_proj += &commitment.shifted_comm.unwrap().mul(cur_challenge);
            }

            cur_challenge = opening_challenges(opening_challenge_counter);
            opening_challenge_counter += 1;
        }

        let mut combined_commitment = combined_commitment_proj.into_affine();

        assert_eq!(proof.hiding_comm.is_some(), proof.rand.is_some());
        if proof.hiding_comm.is_some() {
            let hiding_comm = proof.hiding_comm.unwrap();
            let rand = proof.rand.unwrap();

            let hiding_challenge = Self::compute_random_oracle_challenge(
                &to_bytes![combined_commitment, point, combined_v, hiding_comm].unwrap(),
            );
            combined_commitment_proj += &(hiding_comm.mul(hiding_challenge) - &vk.s.mul(rand));
            combined_commitment = combined_commitment_proj.into_affine();
        }

        // Challenge for each round
        let mut round_challenges = Vec::with_capacity(log_d);
        let mut round_challenge = Self::compute_random_oracle_challenge(
            &to_bytes![combined_commitment, point, combined_v].unwrap(),
        );

        let h_prime = vk.h.mul(round_challenge);

        let mut round_commitment_proj = combined_commitment_proj + &h_prime.mul(&combined_v);

        let l_iter = proof.l_vec.iter();
        let r_iter = proof.r_vec.iter();

        for (l, r) in l_iter.zip(r_iter) {
            round_challenge = Self::compute_random_oracle_challenge(
                &to_bytes![round_challenge, l, r].unwrap(),
            );
            round_challenges.push(round_challenge);
            round_commitment_proj +=
                &(l.mul(round_challenge.inverse().unwrap()) + &r.mul(round_challenge));
        }

        let check_poly = SuccinctCheckPolynomial::<G::ScalarField>(round_challenges);
        let v_prime = check_poly.evaluate(point) * &proof.c;
        let h_prime = h_prime.into_affine();

        let check_commitment_elem: G::Projective = Self::cm_commit(
            &[proof.final_comm_key.clone(), h_prime],
            &[proof.c.clone(), v_prime],
            None,
            None,
        );

        if !ProjectiveCurve::is_zero(&(round_commitment_proj - &check_commitment_elem)) {
            end_timer!(check_time);
            if cfg!(feature = "bench") { return Some(check_poly) } else { return None }
        }

        end_timer!(check_time);
        Some(check_poly)
    }

    /// Perform the succinct check of proof, returning the succinct check polynomial (the xi_s)
    /// and the GFinal.
    fn succinct_batch_check_individual_opening_challenges<'a, R: RngCore>(
        vk: &VerifierKey<G>,
        commitments: impl Clone + IntoIterator<Item = &'a LabeledCommitment<Commitment<G>>>,
        query_set: &QuerySet<G::ScalarField>,
        values: &Evaluations<G::ScalarField>,
        batch_proof: &BatchProof<G>,
        opening_challenges: &dyn Fn(u64) -> G::ScalarField,
        _rng: &mut R,
    ) -> Result<(SuccinctCheckPolynomial<G::ScalarField>, G), Error>
    {
        let batch_check_time = start_timer!(|| "Multi poly multi point batch check: succinct part");

        let mut query_to_labels_map = BTreeMap::new();
        for (label, (point_label, point)) in query_set.iter() {
            let labels = query_to_labels_map
                .entry(point_label)
                .or_insert((point, BTreeSet::new()));
            labels.1.insert(label);
        }

        // v_i values
        let mut v_values = batch_proof.batch_values.clone();

        // y_i values
        let mut y_values = vec![];

        // x_i values
        let mut points = vec![];

        for (_point_label, (point, labels)) in query_to_labels_map.into_iter() {
            for label in labels.into_iter() {
                let y_i = values
                    .get(&(label.clone(), *point))
                    .ok_or(Error::MissingEvaluation {
                        label: label.to_string(),
                    })?;
                y_values.push(*y_i);
                points.push(point);
            }
        }

        // Commitment of the h(X) polynomial
        let batch_commitment = batch_proof.batch_commitment;

        // Fresh random challenge x
        let point = Self::compute_random_oracle_challenge(
            &to_bytes![
                batch_commitment,
                points
            ]
                .unwrap(),
        );

        let mut opening_challenge_counter = 0;

        // lambda
        let mut cur_challenge = opening_challenges(opening_challenge_counter);
        opening_challenge_counter += 1;

        let mut computed_batch_v = G::ScalarField::zero();

        for ((v_i, y_i), x_i) in v_values.clone().into_iter().zip(y_values.clone()).zip(points.clone()) {

            computed_batch_v = computed_batch_v + &(cur_challenge * &((v_i - &y_i) / &(point - x_i)));

            cur_challenge = opening_challenges(opening_challenge_counter);
            opening_challenge_counter += 1;
        }

        // Reconstructed v value added to the check
        v_values.push(computed_batch_v);

        // The commitment to h(X) polynomial added to the check
        let mut commitments = commitments.clone().into_iter().collect::<Vec<&'a LabeledCommitment<Commitment<G>>>>();
        let labeled_batch_commitment = LabeledCommitment::new(
            format!("Batch"),
            Commitment { comm: batch_commitment, shifted_comm: None },
            None
        );
        commitments.push(&labeled_batch_commitment);

        end_timer!(batch_check_time);

        let proof = &batch_proof.proof;

        let check_time = start_timer!(|| "Checking evaluations");
        let d = vk.supported_degree();

        // `log_d` is ceil(log2 (d + 1)), which is the number of steps to compute all of the challenges
        let log_d = algebra::log2(d + 1) as usize;

        if proof.l_vec.len() != proof.r_vec.len() || proof.l_vec.len() != log_d {
            return Err(Error::IncorrectInputLength(
                format!(
                    "Expected proof vectors to be {:}. Instead, l_vec size is {:} and r_vec size is {:}",
                    log_d,
                    proof.l_vec.len(),
                    proof.r_vec.len()
                )
            ));
        }

        let check_poly = Self::succinct_check(vk, commitments, point, v_values, proof, opening_challenges);

        if check_poly.is_none() {
            end_timer!(check_time);
            return Err(Error::FailedSuccinctCheck);
        }

        end_timer!(check_time);

        Ok((check_poly.unwrap(), proof.final_comm_key))
    }

    fn check_degrees_and_bounds(
        supported_degree: usize,
        p: &LabeledPolynomial<G::ScalarField>,
    ) -> Result<(), Error> {
        if p.degree() > supported_degree {
            return Err(Error::TooManyCoefficients {
                num_coefficients: p.degree() + 1,
                num_powers: supported_degree + 1,
            });
        }

        if let Some(bound) = p.degree_bound() {
            if bound < p.degree() || bound > supported_degree {
                return Err(Error::IncorrectDegreeBound {
                    poly_degree: p.degree(),
                    degree_bound: bound,
                    supported_degree,
                    label: p.label().to_string(),
                });
            }
        }

        Ok(())
    }

    fn shift_polynomial(
        ck: &CommitterKey<G>,
        p: &Polynomial<G::ScalarField>,
        degree_bound: usize,
    ) -> Polynomial<G::ScalarField> {
        if p.is_zero() {
            Polynomial::zero()
        } else {
            let mut shifted_polynomial_coeffs =
                vec![G::ScalarField::zero(); ck.supported_degree() - degree_bound];
            shifted_polynomial_coeffs.extend_from_slice(&p.coeffs);
            Polynomial::from_coefficients_vec(shifted_polynomial_coeffs)
        }
    }

    fn combine_shifted_rand(
        combined_rand: Option<G::ScalarField>,
        new_rand: Option<G::ScalarField>,
        coeff: G::ScalarField,
    ) -> Option<G::ScalarField> {
        if let Some(new_rand) = new_rand {
            let coeff_new_rand = new_rand * &coeff;
            return Some(combined_rand.map_or(coeff_new_rand, |r| r + &coeff_new_rand));
        };

        combined_rand
    }

    fn combine_shifted_comm(
        combined_comm: Option<G::Projective>,
        new_comm: Option<G>,
        coeff: G::ScalarField,
    ) -> Option<G::Projective> {
        if let Some(new_comm) = new_comm {
            let coeff_new_comm = new_comm.mul(coeff);
            return Some(combined_comm.map_or(coeff_new_comm, |c| c + &coeff_new_comm));
        };

        combined_comm
    }

    fn construct_labeled_commitments(
        lc_info: &[(String, Option<usize>)],
        elements: &[G::Projective],
    ) -> Vec<LabeledCommitment<Commitment<G>>> {
        let comms = G::Projective::batch_normalization_into_affine(elements.to_vec());
        let mut commitments = Vec::new();

        let mut i = 0;
        for info in lc_info.into_iter() {
            let commitment;
            let label = info.0.clone();
            let degree_bound = info.1;

            if degree_bound.is_some() {
                commitment = Commitment {
                    comm: comms[i].clone(),
                    shifted_comm: Some(comms[i + 1].clone()),
                };

                i += 2;
            } else {
                commitment = Commitment {
                    comm: comms[i].clone(),
                    shifted_comm: None,
                };

                i += 1;
            }

            commitments.push(LabeledCommitment::new(label, commitment, degree_bound));
        }

        return commitments;
    }

    fn sample_generators(num_generators: usize) -> Vec<G> {
        let generators: Vec<_> = (0..num_generators).into_par_iter()
            .map(|i| {
                let i = i as u64;
                let mut hash = D::digest(&to_bytes![&Self::PROTOCOL_NAME, i].unwrap());
                let mut g = G::from_random_bytes(&hash);
                let mut j = 0u64;
                while g.is_none() {
                    hash = D::digest(&to_bytes![&Self::PROTOCOL_NAME, i, j].unwrap());
                    g = G::from_random_bytes(&hash);
                    j += 1;
                }
                let generator = g.unwrap();
                generator.mul_by_cofactor().into_projective()
            })
            .collect();

        G::Projective::batch_normalization_into_affine(generators)
    }

    /// Perform bullet reduce at the commitment last stage
    fn polycommit_round_reduce(
        round_challenge: G::ScalarField,
        round_challenge_inv: G::ScalarField,
        c_l: &mut [G::ScalarField],
        c_r: &[G::ScalarField],
        z_l: &mut [G::ScalarField],
        z_r: &[G::ScalarField],
        k_l: &mut [G::Projective],
        k_r: &[G],
    ) {
        #[cfg(feature = "gpu")]
        if get_gpu_min_length() <= k_l.len() {
            match get_kernels() {
                Ok(kernels) => {
                    match kernels[0].polycommit_round_reduce(
                        round_challenge,
                        round_challenge_inv,
                        c_l,
                        c_r,
                        z_l,
                        z_r,
                        k_l,
                        k_r
                    ) {
                        Ok(_) => {},
                        Err(error) => { panic!("{}", error); }
                    }
                },
                Err(error) => {
                    panic!("{}", error);
                }
            }
            return;
        }

        c_l.par_iter_mut()
            .zip(c_r)
            .for_each(|(c_l, c_r)| *c_l += &(round_challenge_inv * c_r));

        z_l.par_iter_mut()
            .zip(z_r)
            .for_each(|(z_l, z_r)| *z_l += &(round_challenge * z_r));

        k_l.par_iter_mut()
            .zip(k_r)
            .for_each(|(k_l, k_r)| *k_l += &(k_r.mul(round_challenge)));
    }
}

impl<G: AffineCurve, D: Digest> PolynomialCommitment<G::ScalarField> for InnerProductArgPC<G, D> {
    type UniversalParams = UniversalParams<G>;
    type CommitterKey = CommitterKey<G>;
    type VerifierKey = VerifierKey<G>;
    type PreparedVerifierKey = PreparedVerifierKey<G>;
    type Commitment = Commitment<G>;
    type PreparedCommitment = PreparedCommitment<G>;
    type Randomness = Randomness<G>;
    type Proof = Proof<G>;
    type BatchProof = BatchProof<G>;
    type Error = Error;

    fn setup<R: RngCore>(
        max_degree: usize,
        _rng: &mut R,
    ) -> Result<Self::UniversalParams, Self::Error> {
        // Ensure that max_degree + 1 is a power of 2
        let max_degree = (max_degree + 1).next_power_of_two() - 1;

        let setup_time = start_timer!(|| format!("Sampling {} generators", max_degree + 3));
        let mut generators = Self::sample_generators(max_degree + 3);
        end_timer!(setup_time);

        let h = generators.pop().unwrap();
        let s = generators.pop().unwrap();

        let pp = UniversalParams {
            comm_key: generators,
            h,
            s,
        };

        Ok(pp)
    }

    fn trim(
        pp: &Self::UniversalParams,
        supported_degree: usize,
        _supported_hiding_bound: usize,
        _enforced_degree_bounds: Option<&[usize]>,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), Self::Error> {
        // Ensure that supported_degree + 1 is a power of two
        let supported_degree = (supported_degree + 1).next_power_of_two() - 1;
        if supported_degree > pp.max_degree() {
            return Err(Error::TrimmingDegreeTooLarge);
        }

        let trim_time =
            start_timer!(|| format!("Trimming to supported degree of {}", supported_degree));

        let ck = CommitterKey {
            comm_key: pp.comm_key[0..(supported_degree + 1)].to_vec(),
            h: pp.h.clone(),
            s: pp.s.clone(),
            max_degree: pp.max_degree(),
        };

        let vk = VerifierKey {
            comm_key: pp.comm_key[0..(supported_degree + 1)].to_vec(),
            h: pp.h.clone(),
            s: pp.s.clone(),
            max_degree: pp.max_degree(),
        };

        end_timer!(trim_time);

        Ok((ck, vk))
    }

    /// Outputs a commitment to `polynomial`.
    fn commit<'a>(
        ck: &Self::CommitterKey,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField>>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<
        (
            Vec<LabeledCommitment<Self::Commitment>>,
            Vec<Self::Randomness>,
        ),
        Self::Error,
    > {
        let rng = &mut crate::optional_rng::OptionalRng(rng);
        let mut comms = Vec::new();
        let mut rands = Vec::new();

        let commit_time = start_timer!(|| "Committing to polynomials");
        for labeled_polynomial in polynomials {
            Self::check_degrees_and_bounds(ck.supported_degree(), labeled_polynomial)?;

            let polynomial = labeled_polynomial.polynomial();
            let label = labeled_polynomial.label();
            let hiding_bound = labeled_polynomial.hiding_bound();
            let degree_bound = labeled_polynomial.degree_bound();

            let commit_time = start_timer!(|| format!(
                "Polynomial {} of degree {}, degree bound {:?}, and hiding bound {:?}",
                label,
                polynomial.degree(),
                degree_bound,
                hiding_bound,
            ));

            let randomness = if let Some(h) = hiding_bound {
                Randomness::rand(h, degree_bound.is_some(), rng)
            } else {
                Randomness::empty()
            };

            let comm = Self::cm_commit(
                &ck.comm_key[..(polynomial.degree() + 1)],
                &polynomial.coeffs,
                Some(ck.s),
                Some(randomness.rand),
            )
            .into_affine();

            let shifted_comm = degree_bound.map(|d| {
                Self::cm_commit(
                    &ck.comm_key[(ck.supported_degree() - d)..],
                    &polynomial.coeffs,
                    Some(ck.s),
                    randomness.shifted_rand,
                )
                .into_affine()
            });

            let commitment = Commitment { comm, shifted_comm };
            let labeled_comm = LabeledCommitment::new(label.to_string(), commitment, degree_bound);

            comms.push(labeled_comm);
            rands.push(randomness);

            end_timer!(commit_time);
        }

        end_timer!(commit_time);
        Ok((comms, rands))
    }

    fn open_individual_opening_challenges<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: G::ScalarField,
        opening_challenges: &dyn Fn(u64) -> G::ScalarField,
        rands: impl IntoIterator<Item = &'a Self::Randomness>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Self::Error>
    where
        Self::Commitment: 'a,
        Self::Randomness: 'a,
    {
        let mut combined_polynomial = Polynomial::zero();
        let mut combined_rand = G::ScalarField::zero();
        let mut combined_commitment_proj = <G::Projective as ProjectiveCurve>::zero();

        let mut has_hiding = false;

        let polys_iter = labeled_polynomials.into_iter();
        let rands_iter = rands.into_iter();
        let comms_iter = commitments.into_iter();

        let combine_time = start_timer!(|| "Combining polynomials, randomness, and commitments.");

        let mut opening_challenge_counter = 0;
        let mut cur_challenge = opening_challenges(opening_challenge_counter);
        opening_challenge_counter += 1;

        for (labeled_polynomial, (labeled_commitment, randomness)) in
            polys_iter.zip(comms_iter.zip(rands_iter))
        {
            let label = labeled_polynomial.label();
            assert_eq!(labeled_polynomial.label(), labeled_commitment.label());
            Self::check_degrees_and_bounds(ck.supported_degree(), labeled_polynomial)?;

            let polynomial = labeled_polynomial.polynomial();
            let degree_bound = labeled_polynomial.degree_bound();
            let hiding_bound = labeled_polynomial.hiding_bound();
            let commitment = labeled_commitment.commitment();

            combined_polynomial += (cur_challenge, polynomial);
            combined_commitment_proj += &commitment.comm.mul(cur_challenge);

            if hiding_bound.is_some() {
                has_hiding = true;
                combined_rand += &(cur_challenge * &randomness.rand);
            }

            cur_challenge = opening_challenges(opening_challenge_counter);
            opening_challenge_counter += 1;

            let has_degree_bound = degree_bound.is_some();

            assert_eq!(
                has_degree_bound,
                commitment.shifted_comm.is_some(),
                "shifted_comm mismatch for {}",
                label
            );

            assert_eq!(
                degree_bound,
                labeled_commitment.degree_bound(),
                "labeled_comm degree bound mismatch for {}",
                label
            );
            if let Some(degree_bound) = degree_bound {
                let shifted_polynomial = Self::shift_polynomial(ck, polynomial, degree_bound);
                combined_polynomial += (cur_challenge, &shifted_polynomial);
                combined_commitment_proj += &commitment.shifted_comm.unwrap().mul(cur_challenge);

                if hiding_bound.is_some() {
                    let shifted_rand = randomness.shifted_rand;
                    assert!(
                        shifted_rand.is_some(),
                        "shifted_rand.is_none() for {}",
                        label
                    );
                    combined_rand += &(cur_challenge * &shifted_rand.unwrap());
                }
            }

            cur_challenge = opening_challenges(opening_challenge_counter);
            opening_challenge_counter += 1;
        }

        end_timer!(combine_time);

        let combined_v = combined_polynomial.evaluate(point);

        // Pad the coefficients to the appropriate vector size
        let d = ck.supported_degree();

        // `log_d` is ceil(log2 (d + 1)), which is the number of steps to compute all of the challenges
        let log_d = algebra::log2(d + 1) as usize;

        let mut combined_commitment;
        let mut hiding_commitment = None;

        if has_hiding {
            let mut rng = rng.expect("hiding commitments require randomness");
            let hiding_time = start_timer!(|| "Applying hiding.");
            let mut hiding_polynomial = Polynomial::rand(d, &mut rng);
            hiding_polynomial -=
                &Polynomial::from_coefficients_slice(&[hiding_polynomial.evaluate(point)]);

            let hiding_rand = G::ScalarField::rand(rng);
            let hiding_commitment_proj = Self::cm_commit(
                ck.comm_key.as_slice(),
                hiding_polynomial.coeffs.as_slice(),
                Some(ck.s),
                Some(hiding_rand),
            );

            let mut batch = G::Projective::batch_normalization_into_affine(vec![
                combined_commitment_proj,
                hiding_commitment_proj,
            ]);
            hiding_commitment = Some(batch.pop().unwrap());
            combined_commitment = batch.pop().unwrap();

            let hiding_challenge = Self::compute_random_oracle_challenge(
                &to_bytes![
                    combined_commitment,
                    point,
                    combined_v,
                    hiding_commitment.unwrap()
                ]
                .unwrap(),
            );
            combined_polynomial += (hiding_challenge, &hiding_polynomial);
            combined_rand += &(hiding_challenge * &hiding_rand);
            combined_commitment_proj +=
                &(hiding_commitment_proj.mul(&hiding_challenge) - &ck.s.mul(combined_rand));

            end_timer!(hiding_time);
        }

        let combined_rand = if has_hiding {
            Some(combined_rand)
        } else {
            None
        };

        let proof_time =
            start_timer!(|| format!("Generating proof for degree {} combined polynomial", d + 1));

        combined_commitment = combined_commitment_proj.into_affine();

        // ith challenge
        let mut round_challenge = Self::compute_random_oracle_challenge(
            &to_bytes![combined_commitment, point, combined_v].unwrap(),
        );

        let h_prime = ck.h.mul(round_challenge).into_affine();

        // Pads the coefficients with zeroes to get the number of coeff to be d+1
        let mut coeffs = combined_polynomial.coeffs;
        if coeffs.len() < d + 1 {
            for _ in coeffs.len()..(d + 1) {
                coeffs.push(G::ScalarField::zero());
            }
        }
        let mut coeffs = coeffs.as_mut_slice();

        // Powers of z
        let mut z: Vec<G::ScalarField> = Vec::with_capacity(d + 1);
        let mut cur_z: G::ScalarField = G::ScalarField::one();
        for _ in 0..(d + 1) {
            z.push(cur_z);
            cur_z *= &point;
        }
        let mut z = z.as_mut_slice();

        // This will be used for transforming the key in each step
        let mut key_proj: Vec<G::Projective> = ck.comm_key.iter().map(|x| (*x).into_projective()).collect();
        let mut key_proj = key_proj.as_mut_slice();

        let mut temp;

        // Key for MSM
        // We initialize this to capacity 0 initially because we want to use the key slice first
        let mut comm_key = &ck.comm_key;

        let mut l_vec = Vec::with_capacity(log_d);
        let mut r_vec = Vec::with_capacity(log_d);

        let mut n = d + 1;
        while n > 1 {
            let (coeffs_l, coeffs_r) = coeffs.split_at_mut(n / 2);
            let (z_l, z_r) = z.split_at_mut(n / 2);
            let (key_l, key_r) = comm_key.split_at(n / 2);
            let (key_proj_l, _) = key_proj.split_at_mut(n / 2);

            let l = Self::cm_commit(key_l, coeffs_r, None, None)
                + &h_prime.mul(Self::inner_product(coeffs_r, z_l));

            let r = Self::cm_commit(key_r, coeffs_l, None, None)
                + &h_prime.mul(Self::inner_product(coeffs_l, z_r));

            let lr = G::Projective::batch_normalization_into_affine(vec![l, r]);
            l_vec.push(lr[0]);
            r_vec.push(lr[1]);

            round_challenge = Self::compute_random_oracle_challenge(
                &to_bytes![round_challenge, lr[0], lr[1]].unwrap(),
            );
            let round_challenge_inv = round_challenge.inverse().unwrap();

            Self::polycommit_round_reduce(
                round_challenge,
                round_challenge_inv,
                coeffs_l,
                coeffs_r,
                z_l,
                z_r,
                key_proj_l,
                key_r,
            );

            coeffs = coeffs_l;
            z = z_l;

            key_proj = key_proj_l;
            temp = G::Projective::batch_normalization_into_affine(key_proj.to_vec());
            comm_key = &temp;

            n /= 2;
        }

        end_timer!(proof_time);

        Ok(Proof {
            l_vec,
            r_vec,
            final_comm_key: comm_key[0],
            c: coeffs[0],
            hiding_comm: hiding_commitment,
            rand: combined_rand,
        })
    }

    /// Refers to the "Multi-poly multi-point assertions" section of the "Recursive SNARKs based on Marlin" document
    fn batch_open_individual_opening_challenges<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl Clone + IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField>>,
        commitments: impl Clone + IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<G::ScalarField>,
        opening_challenges: &dyn Fn(u64) -> G::ScalarField,
        rands: impl Clone + IntoIterator<Item = &'a Self::Randomness>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::BatchProof, Self::Error>
    where
        Self::Randomness: 'a,
        Self::Commitment: 'a,
    {
        let batch_time = start_timer!(|| "Multi poly multi point batching.");

        let mut opening_challenge_counter = 0;
        
        // lambda
        let mut cur_challenge = opening_challenges(opening_challenge_counter);
        opening_challenge_counter += 1;

        let mut query_to_labels_map = BTreeMap::new();

        let poly_map: BTreeMap<_, _> = labeled_polynomials
            .clone()
            .into_iter()
            .map(|poly| (poly.label(), poly))
            .collect();

        for (label, (point_label, point)) in query_set.iter() {
            let labels = query_to_labels_map
                .entry(point_label)
                .or_insert((point, BTreeSet::new()));
            labels.1.insert(label);
        }

        let mut points = vec![];

        // h(X)
        let mut batch_polynomial = Polynomial::zero();

        for (
            _point_label, (point, labels),
        ) in query_to_labels_map.into_iter() {
            for label in labels {
                let labeled_polynomial =
                    poly_map.get(label).ok_or(Error::MissingPolynomial {
                        label: label.to_string(),
                    })?;

                points.push(point);

                // y_i
                let evaluated_y = labeled_polynomial.polynomial().evaluate(*point);

                // (p_i(X) - y_i) / (X - x_i)
                let polynomial =
                    &(labeled_polynomial.polynomial() - &Polynomial::from_coefficients_vec(vec![evaluated_y]))
                        /
                        &Polynomial::from_coefficients_vec(vec![
                            (G::ScalarField::zero() - point),
                            G::ScalarField::one()
                        ]);

                // h(X) = SUM( lambda^i * ((p_i(X) - y_i) / (X - x_i)) )
                batch_polynomial += (cur_challenge, &polynomial);

                // lambda^i
                cur_challenge = opening_challenges(opening_challenge_counter);
                opening_challenge_counter += 1;
            }
        }

        // Commitment of the h(X) polynomial
        let batch_commitment = Self::cm_commit(
            ck.comm_key.as_slice(),
            batch_polynomial.coeffs.as_slice(),
            None,
            None,
        ).into_affine();

        // Fresh random challenge x
        let point= Self::compute_random_oracle_challenge(
            &to_bytes![
                batch_commitment,
                points                
            ]
            .unwrap(),
        );

        // Values: v_i = p_i(x), where x is fresh random challenge
        let mut batch_values = vec![];
        for labeled_polynomial in labeled_polynomials.clone().into_iter() {
            batch_values.push(labeled_polynomial.polynomial().evaluate(point));
        }

        // h(X) polynomial added to the set of polynomials for multi-poly single-point batching
        let mut labeled_polynomials = labeled_polynomials.clone().into_iter().collect::<Vec<&'a LabeledPolynomial<G::ScalarField>>>();
        let labeled_batch_polynomial = LabeledPolynomial::new(
            format!("Batch"),
            batch_polynomial,
            None,
            None
        );
        labeled_polynomials.push(&labeled_batch_polynomial);

        // Commitment of h(X) polynomial added to the set of polynomials for multi-poly single-point batching
        let mut commitments = commitments.clone().into_iter().collect::<Vec<&'a LabeledCommitment<Self::Commitment>>>();
        let labeled_batch_commitment = LabeledCommitment::new(
            format!("Batch"),
            Commitment { comm: batch_commitment, shifted_comm: None },
            None
        );
        commitments.push(&labeled_batch_commitment);

        let mut rands = rands.clone().into_iter().collect::<Vec<&'a Self::Randomness>>();
        let batch_randomness = Randomness::empty();
        rands.push(&batch_randomness);

        end_timer!(batch_time);

        let proof = Self::open_individual_opening_challenges(
            ck,
            labeled_polynomials,
            commitments,
            point,
            opening_challenges,
            rands,
            rng
        )?;

        Ok(BatchProof {
            proof,
            batch_commitment,
            batch_values
        })
    }

    fn check_individual_opening_challenges<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: G::ScalarField,
        values: impl IntoIterator<Item = G::ScalarField>,
        proof: &Self::Proof,
        opening_challenges: &dyn Fn(u64) -> G::ScalarField,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        let check_time = start_timer!(|| "Checking evaluations");
        let d = vk.supported_degree();

        // `log_d` is ceil(log2 (d + 1)), which is the number of steps to compute all of the challenges
        let log_d = algebra::log2(d + 1) as usize;

        if proof.l_vec.len() != proof.r_vec.len() || proof.l_vec.len() != log_d {
            return Err(Error::IncorrectInputLength(
                format!(
                    "Expected proof vectors to be {:}. Instead, l_vec size is {:} and r_vec size is {:}",
                    log_d,
                    proof.l_vec.len(),
                    proof.r_vec.len()
                )
            ));
        }

        let check_poly =
            Self::succinct_check(vk, commitments, point, values, proof, opening_challenges);

        if check_poly.is_none() {
            return Ok(false);
        }

        let check_poly_time = start_timer!(|| "Compute check poly");
        let check_poly_coeffs = check_poly.unwrap().compute_coeffs();
        end_timer!(check_poly_time);

        let hard_time = start_timer!(|| "DLOG hard part");
        let final_key = Self::cm_commit(
            vk.comm_key.as_slice(),
            check_poly_coeffs.as_slice(),
            None,
            None,
        );
        end_timer!(hard_time);

        if !ProjectiveCurve::is_zero(&(final_key - &proof.final_comm_key.into_projective())) {
            end_timer!(check_time);
            return Ok(false);
        }

        end_timer!(check_time);
        Ok(true)
    }

    fn batch_check_individual_opening_challenges<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        commitments: impl Clone + IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<G::ScalarField>,
        evaluations: &Evaluations<G::ScalarField>,
        batch_proof: &Self::BatchProof,
        opening_challenges: &dyn Fn(u64) -> G::ScalarField,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        // DLOG "succinct" part
        let (check_poly, proof_final_key) = Self::succinct_batch_check_individual_opening_challenges(
            vk,
            commitments,
            query_set,
            evaluations,
            batch_proof,
            opening_challenges,
            rng
        )?;

        // DLOG hard part
        let check_time = start_timer!(|| "DLOG hard part");
        let check_poly_coeffs = check_poly.compute_coeffs();
        let final_key = Self::cm_commit(
            vk.comm_key.as_slice(),
            check_poly_coeffs.as_slice(),
            None,
            None,
        );
        if !ProjectiveCurve::is_zero(&(final_key - &proof_final_key.into_projective())) {
            end_timer!(check_time);
            return Ok(false);
        }

        end_timer!(check_time);
        Ok(true)
    }

    /// Batch verification of multiple Self::BatchProof(s)
    fn batch_check_batch_proofs<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a [LabeledCommitment<Self::Commitment>]>,
        query_sets: impl IntoIterator<Item = &'a QuerySet<'a, G::ScalarField>>,
        values: impl IntoIterator<Item = &'a Evaluations<'a, G::ScalarField>>,
        proofs: impl IntoIterator<Item = &'a Self::BatchProof>,
        opening_challenges: impl IntoIterator<Item = G::ScalarField>,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
        where
            Self::Commitment: 'a,
            Self::BatchProof: 'a,
    {
        let comms = commitments.into_iter().collect::<Vec<_>>();
        let query_sets = query_sets.into_iter().collect::<Vec<_>>();
        let values = values.into_iter().collect::<Vec<_>>();
        let proofs = proofs.into_iter().collect::<Vec<_>>();
        let opening_challenges = opening_challenges.into_iter().collect::<Vec<_>>();

        let succinct_time = start_timer!(|| format!("Succinct verification of proofs"));

        let xi_s_and_final_comm_keys = comms.into_par_iter()
            .zip(query_sets)
            .zip(values)
            .zip(proofs)
            .zip(opening_challenges)
            .map(|((((commitments, query_set), values), proof), opening_challenge)|
                {
                    // Perform succinct check of i-th proof
                    let opening_challenge_f = |pow| opening_challenge.pow(&[pow]);
                    let (challenges, final_comm_key) = Self::succinct_batch_check_individual_opening_challenges(
                        vk,
                        commitments,
                        query_set,
                        values,
                        proof,
                        &opening_challenge_f,
                        &mut OsRng::default() // the rng doesn't matter
                    ).unwrap();

                    (challenges, final_comm_key)
                }
            ).collect::<Vec<_>>();

        let final_comm_keys = xi_s_and_final_comm_keys.iter().map(|(_, key)| key.clone()).collect::<Vec<_>>();
        let xi_s_vec = xi_s_and_final_comm_keys.into_iter().map(|(chal, _)| chal).collect::<Vec<_>>();

        end_timer!(succinct_time);

        let batching_time = start_timer!(|| "Combine check polynomials and final comm keys");

        // Sample batching challenge
        let random_scalar = G::ScalarField::rand(rng);
        let mut batching_chal = G::ScalarField::one();

        // Collect the powers of the batching challenge in a vector
        let mut batching_chal_pows = vec![G::ScalarField::zero(); xi_s_vec.len()];
        for i in 0..batching_chal_pows.len() {
            batching_chal_pows[i] = batching_chal;
            batching_chal *= &random_scalar;
        }

        // Compute the combined_check_poly
        let combined_check_poly = batching_chal_pows
            .par_iter()
            .zip(xi_s_vec)
            .map(|(&chal, xi_s)| {
                Polynomial::from_coefficients_vec(xi_s.compute_scaled_coeffs(-chal))
            }).reduce(|| Polynomial::zero(), |acc, scaled_poly| &acc + &scaled_poly);
        end_timer!(batching_time);

        // DLOG hard part.
        // The equation to check would be:
        // lambda_1 * gfin_1 + ... + lambda_n * gfin_n - combined_h_1 * g_vk_1 - ... - combined_h_m * g_vk_m = 0
        // Where combined_h_i = lambda_1 * h_1_i + ... + lambda_n * h_n_i
        // We do final verification and the batching of the GFin in a single MSM
        let hard_time = start_timer!(|| "Batch verify hard parts");
        let final_val = Self::cm_commit(
            &[final_comm_keys.as_slice(), vk.comm_key.as_slice()].concat(),
            &[batching_chal_pows.as_slice(), combined_check_poly.coeffs.as_slice()].concat(),
            None,
            None,
        );
        if !ProjectiveCurve::is_zero(&final_val) {
            end_timer!(hard_time);
            return Ok(false);
        }
        end_timer!(hard_time);
        Ok(true)
    }

    fn open_combinations_individual_opening_challenges<'a>(
        ck: &Self::CommitterKey,
        lc_s: impl IntoIterator<Item = &'a LinearCombination<G::ScalarField>>,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<G::ScalarField>,
        opening_challenges: &dyn Fn(u64) -> G::ScalarField,
        rands: impl IntoIterator<Item = &'a Self::Randomness>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<BatchLCProof<G::ScalarField, Self>, Self::Error>
    where
        Self::Randomness: 'a,
        Self::Commitment: 'a,
    {
        let label_poly_map = polynomials
            .into_iter()
            .zip(rands)
            .zip(commitments)
            .map(|((p, r), c)| (p.label(), (p, r, c)))
            .collect::<BTreeMap<_, _>>();

        let mut lc_polynomials = Vec::new();
        let mut lc_randomness = Vec::new();
        let mut lc_commitments = Vec::new();
        let mut lc_info = Vec::new();

        for lc in lc_s {
            let lc_label = lc.label().clone();
            let mut poly = Polynomial::zero();
            let mut degree_bound = None;
            let mut hiding_bound = None;

            let mut combined_comm = <G::Projective as ProjectiveCurve>::zero();
            let mut combined_shifted_comm: Option<G::Projective> = None;

            let mut combined_rand = G::ScalarField::zero();
            let mut combined_shifted_rand: Option<G::ScalarField> = None;

            let num_polys = lc.len();
            for (coeff, label) in lc.iter().filter(|(_, l)| !l.is_one()) {
                let label: &String = label.try_into().expect("cannot be one!");
                let &(cur_poly, cur_rand, cur_comm) =
                    label_poly_map.get(label).ok_or(Error::MissingPolynomial {
                        label: label.to_string(),
                    })?;

                if num_polys == 1 && cur_poly.degree_bound().is_some() {
                    assert!(
                        coeff.is_one(),
                        "Coefficient must be one for degree-bounded equations"
                    );
                    degree_bound = cur_poly.degree_bound();
                } else if cur_poly.degree_bound().is_some() {
                    eprintln!("Degree bound when number of equations is non-zero");
                    return Err(Self::Error::EquationHasDegreeBounds(lc_label));
                }

                // Some(_) > None, always.
                hiding_bound = std::cmp::max(hiding_bound, cur_poly.hiding_bound());
                poly += (*coeff, cur_poly.polynomial());

                combined_rand += &(cur_rand.rand * coeff);
                combined_shifted_rand = Self::combine_shifted_rand(
                    combined_shifted_rand,
                    cur_rand.shifted_rand,
                    *coeff,
                );

                let commitment = cur_comm.commitment();
                combined_comm += &commitment.comm.mul(*coeff);
                combined_shifted_comm = Self::combine_shifted_comm(
                    combined_shifted_comm,
                    commitment.shifted_comm,
                    *coeff,
                );
            }

            let lc_poly =
                LabeledPolynomial::new(lc_label.clone(), poly, degree_bound, hiding_bound);
            lc_polynomials.push(lc_poly);
            lc_randomness.push(Randomness {
                rand: combined_rand,
                shifted_rand: combined_shifted_rand,
            });

            lc_commitments.push(combined_comm);
            if let Some(combined_shifted_comm) = combined_shifted_comm {
                lc_commitments.push(combined_shifted_comm);
            }

            lc_info.push((lc_label, degree_bound));
        }

        let lc_commitments = Self::construct_labeled_commitments(&lc_info, &lc_commitments);

        let proof = Self::batch_open_individual_opening_challenges(
            ck,
            lc_polynomials.iter(),
            lc_commitments.iter(),
            &query_set,
            opening_challenges,
            lc_randomness.iter(),
            rng,
        )?;
        Ok(BatchLCProof { proof, evals: None })
    }

    /// Checks that `values` are the true evaluations at `query_set` of the polynomials
    /// committed in `labeled_commitments`.
    fn check_combinations_individual_opening_challenges<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        lc_s: impl IntoIterator<Item = &'a LinearCombination<G::ScalarField>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<G::ScalarField>,
        evaluations: &Evaluations<G::ScalarField>,
        proof: &BatchLCProof<G::ScalarField, Self>,
        opening_challenges: &dyn Fn(u64) -> G::ScalarField,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        let BatchLCProof { proof, .. } = proof;
        let label_comm_map = commitments
            .into_iter()
            .map(|c| (c.label(), c))
            .collect::<BTreeMap<_, _>>();

        let mut lc_commitments = Vec::new();
        let mut lc_info = Vec::new();
        let mut evaluations = evaluations.clone();
        for lc in lc_s {
            let lc_label = lc.label().clone();
            let num_polys = lc.len();

            let mut degree_bound = None;
            let mut combined_comm = <G::Projective as ProjectiveCurve>::zero();
            let mut combined_shifted_comm: Option<G::Projective> = None;

            for (coeff, label) in lc.iter() {
                if label.is_one() {
                    for (&(ref label, _), ref mut eval) in evaluations.iter_mut() {
                        if label == &lc_label {
                            **eval -= coeff;
                        }
                    }
                } else {
                    let label: &String = label.try_into().unwrap();
                    let &cur_comm = label_comm_map.get(label).ok_or(Error::MissingPolynomial {
                        label: label.to_string(),
                    })?;

                    if num_polys == 1 && cur_comm.degree_bound().is_some() {
                        assert!(
                            coeff.is_one(),
                            "Coefficient must be one for degree-bounded equations"
                        );
                        degree_bound = cur_comm.degree_bound();
                    } else if cur_comm.degree_bound().is_some() {
                        return Err(Self::Error::EquationHasDegreeBounds(lc_label));
                    }

                    let commitment = cur_comm.commitment();
                    combined_comm += &commitment.comm.mul(*coeff);
                    combined_shifted_comm = Self::combine_shifted_comm(
                        combined_shifted_comm,
                        commitment.shifted_comm,
                        *coeff,
                    );
                }
            }

            lc_commitments.push(combined_comm);

            if let Some(combined_shifted_comm) = combined_shifted_comm {
                lc_commitments.push(combined_shifted_comm);
            }

            lc_info.push((lc_label, degree_bound));
        }

        let lc_commitments = Self::construct_labeled_commitments(&lc_info, &lc_commitments);

        Self::batch_check_individual_opening_challenges(
            vk,
            &lc_commitments,
            &query_set,
            &evaluations,
            proof,
            opening_challenges,
            rng,
        )
    }
}

impl<
    G: AffineCurve,
    D: Digest
> PublicAccumulationScheme<G::ScalarField, DLogAccumulator<G>> for InnerProductArgPC<G, D>{

    fn succinct_verify<'a, R: RngCore>(
        vk:                 &Self::VerifierKey,
        commitments:        impl IntoIterator<Item = &'a [LabeledCommitment<Self::Commitment>]>,
        query_sets:         impl IntoIterator<Item = &'a QuerySet<'a, G::ScalarField>>,
        values:             impl IntoIterator<Item = &'a Evaluations<'a, G::ScalarField>>,
        proofs:             impl IntoIterator<Item = &'a Self::BatchProof>,
        opening_challenges: impl IntoIterator<Item = G::ScalarField>,
        _rng:                &mut R,
    ) -> Result<Vec<DLogAccumulator<G>>, Self::Error>
        where
            Self::Commitment: 'a,
            Self::BatchProof: 'a
    {
        let comms = commitments.into_iter().collect::<Vec<_>>();
        let query_sets = query_sets.into_iter().collect::<Vec<_>>();
        let values = values.into_iter().collect::<Vec<_>>();
        let proofs = proofs.into_iter().collect::<Vec<_>>();
        let opening_challenges = opening_challenges.into_iter().collect::<Vec<_>>();

        // Perform succinct verification of all the proofs and collect
        // the xi_s and the GFinal_s into DLogAccumulators
        let succinct_time = start_timer!(|| "Succinct verification of proofs");

        let accumulators = comms.into_par_iter()
            .zip(query_sets)
            .zip(values)
            .zip(proofs)
            .zip(opening_challenges)
            .map(|((((commitments, query_set), values), proof), opening_challenge)|
                {
                    // Perform succinct check of i-th proof
                    let opening_challenge_f = |pow| opening_challenge.pow(&[pow]);
                    let (challenges, final_comm_key) = Self::succinct_batch_check_individual_opening_challenges(
                        vk,
                        commitments,
                        query_set,
                        values,
                        proof,
                        &opening_challenge_f,
                        &mut OsRng::default() // the rng doesn't matter
                    ).unwrap();

                    DLogAccumulator::<G>{
                        g_final: Commitment::<G> { comm: final_comm_key, shifted_comm: None },
                        xi_s: challenges
                    }
                }
            ).collect::<Vec<_>>();
        end_timer!(succinct_time);

        Ok(accumulators)
    }

    fn accumulate<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        vk_hash: Vec<u8>,
        accumulators: Vec<DLogAccumulator<G>>,
        rng: &mut R,
    ) -> Result<Self::Proof, Self::Error>
    {
        let accumulate_time = start_timer!(|| "Accumulate");

        let poly_time = start_timer!(|| "Compute Bullet Polys and their evaluations");
        // Sample a new challenge z
        let z = Self::compute_random_oracle_challenge(
            &to_bytes![vk_hash, accumulators.as_slice()].unwrap(),
        );

        let polys_comms_values = accumulators
            .into_par_iter()
            .enumerate()
            .map(|(i, acc)| {
                let final_comm_key = acc.g_final.comm.clone();
                let xi_s = acc.xi_s;

                // Create a LabeledCommitment out of the g_final
                let labeled_comm = {
                    let comm = Commitment {
                        comm: final_comm_key,
                        shifted_comm: None
                    };

                    LabeledCommitment::new(
                        format!("check_poly_{}", i),
                        comm,
                        None,
                    )
                };

                // Compute the evaluation of the Bullet polynomial at z starting from the xi_s
                let eval = xi_s.evaluate(z);

                // Compute the coefficients of the Bullet polynomial starting from the xi_s
                let check_poly = Polynomial::from_coefficients_vec(xi_s.compute_coeffs());

                (check_poly, labeled_comm, eval)
            }).collect::<Vec<_>>();

        let len = polys_comms_values.len();

        // Sample new opening challenge
        let opening_challenge = Self::compute_random_oracle_challenge(
            &polys_comms_values.iter().flat_map(|(_, _, val)| to_bytes!(val).unwrap()).collect::<Vec<_>>()
        );
        let opening_challenges = |pow| opening_challenge.pow(&[pow]);

        // Save comms and polys into separate vectors
        let comms = polys_comms_values.iter().map(|(_, comm, _)| comm.clone()).collect::<Vec<_>>();
        let polys = polys_comms_values.into_iter().enumerate().map(|(i, (poly, _, _))|
            LabeledPolynomial::new(
                format!("check_poly_{}", i),
                poly,
                None,
                None
            )
        ).collect::<Vec<_>>();
        end_timer!(poly_time);

        let open_time = start_timer!(|| "Produce opening proof for Bullet polys and GFin s");
        // Compute and return opening proof
        let result = Self::open_individual_opening_challenges(
            vk,
            polys.iter(),
            comms.iter(),
            z,
            &opening_challenges,
            vec![Randomness::<G>::empty(); len].iter(),
            Some(rng)
        ).unwrap();
        end_timer!(open_time);

        end_timer!(accumulate_time);

        Ok(result)
    }

    fn verify_accumulation<R: RngCore>(
        vk: &Self::VerifierKey,
        vk_hash: Vec<u8>,
        accumulators: Vec<DLogAccumulator<G>>,
        proof: Self::Proof,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    {
        let check_acc_time = start_timer!(|| "Verify Accumulation");

        let poly_time = start_timer!(|| "Compute Bullet Polys evaluations");
        // Sample a new challenge z
        let z = Self::compute_random_oracle_challenge(
            &to_bytes![vk_hash, accumulators.as_slice()].unwrap(),
        );

        let comms_values = accumulators
            .into_par_iter()
            .enumerate()
            .map(|(i, acc)| {
                let final_comm_key = acc.g_final.comm.clone();
                let xi_s = acc.xi_s;

                // Create a LabeledCommitment out of the g_final
                let labeled_comm = {
                    let comm = Commitment {
                        comm: final_comm_key,
                        shifted_comm: None
                    };

                    LabeledCommitment::new(
                        format!("check_poly_{}", i),
                        comm,
                        None,
                    )
                };

                // Compute the evaluation of the Bullet polynomial at z starting from the xi_s
                let eval = xi_s.evaluate(z);

                (labeled_comm, eval)
            }).collect::<Vec<_>>();

        // Save the evaluations into a separate vec
        let values = comms_values.iter().map(|(_, val)| val.clone()).collect::<Vec<_>>();

        // Sample new opening challenge
        let opening_challenge = Self::compute_random_oracle_challenge(
            &values.iter().flat_map(|val| to_bytes!(val).unwrap()).collect::<Vec<_>>()
        );
        let opening_challenges = |pow| opening_challenge.pow(&[pow]);

        // Save comms into a separate vector
        let comms = comms_values.into_iter().map(|(comm, _)| comm).collect::<Vec<_>>();

        end_timer!(poly_time);

        let check_time = start_timer!(|| "Verify opening proof for Bullet polys evals and GFin s");

        // Check dlog_accumulator
        let result = Self::check_individual_opening_challenges(
            vk,
            comms.iter(),
            z,
            values,
            &proof,
            &opening_challenges,
            Some(rng)
        ).unwrap();
        end_timer!(check_time);

        end_timer!(check_acc_time);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_camel_case_types)]

    use super::InnerProductArgPC;

    use algebra::curves::tweedle::dee::{
        Affine, Projective,
    };
    use blake2::Blake2s;
    use crate::ipa_pc::DLogAccumulator;

    type PC<E, D> = InnerProductArgPC<E, D>;
    type PC_DEE = PC<Affine, Blake2s>;

    #[test]
    fn constant_poly_test() {
        use crate::tests::*;
        constant_poly_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
    }

    #[test]
    fn single_poly_test() {
        use crate::tests::*;
        single_poly_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
    }

    #[test]
    fn quadratic_poly_degree_bound_multiple_queries_test() {
        use crate::tests::*;
        quadratic_poly_degree_bound_multiple_queries_test::<_, PC_DEE>()
            .expect("test failed for tweedle_dee-blake2s");
    }

    #[test]
    fn linear_poly_degree_bound_test() {
        use crate::tests::*;
        linear_poly_degree_bound_test::<_, PC_DEE>()
            .expect("test failed for tweedle_dee-blake2s");
    }

    #[test]
    fn single_poly_degree_bound_test() {
        use crate::tests::*;
        single_poly_degree_bound_test::<_, PC_DEE>()
            .expect("test failed for tweedle_dee-blake2s");
    }

    #[test]
    fn single_poly_degree_bound_multiple_queries_test() {
        use crate::tests::*;
        single_poly_degree_bound_multiple_queries_test::<_, PC_DEE>()
            .expect("test failed for tweedle_dee-blake2s");
    }

    #[test]
    fn two_polys_degree_bound_single_query_test() {
        use crate::tests::*;
        two_polys_degree_bound_single_query_test::<_, PC_DEE>()
            .expect("test failed for tweedle_dee-blake2s");
    }

    #[test]
    fn full_end_to_end_test() {
        use crate::tests::*;
        full_end_to_end_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    #[test]
    fn batch_check_batch_proofs_test() {
        use crate::tests::*;
        batch_check_batch_proofs_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    #[test]
    fn dlog_accumulation_test() {
        use crate::tests::*;
        accumulation_test::<_, DLogAccumulator<Affine>, PC_DEE, Blake2s>().expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    #[test]
    fn single_equation_test() {
        use crate::tests::*;
        single_equation_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    #[test]
    fn two_equation_test() {
        use crate::tests::*;
        two_equation_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    #[test]
    fn two_equation_degree_bound_test() {
        use crate::tests::*;
        two_equation_degree_bound_test::<_, PC_DEE>()
            .expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    #[test]
    fn full_end_to_end_equation_test() {
        use crate::tests::*;
        full_end_to_end_equation_test::<_, PC_DEE>()
            .expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    #[test]
    #[should_panic]
    fn bad_degree_bound_test() {
        use crate::tests::*;
        bad_degree_bound_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    #[test]
    fn polycommit_round_reduce_test() {
        use algebra::fields::tweedle::fr::Fr;
        use algebra::{UniformRand, AffineCurve, ProjectiveCurve, Field};
        use rayon::prelude::*;

        let mut rng = &mut rand::thread_rng();

        let round_challenge = Fr::rand(&mut rng);
        let round_challenge_inv = round_challenge.inverse().unwrap();

        let samples = 1 << 10;

        let mut coeffs_l = (0..samples)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let coeffs_r = (0..samples)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut z_l = (0..samples)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let z_r= (0..samples)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut key_proj_l= (0..samples)
            .map(|_| Projective::rand(&mut rng))
            .collect::<Vec<_>>();

        let key_r= (0..samples)
            .map(|_| Projective::rand(&mut rng).into_affine())
            .collect::<Vec<_>>();

        let mut gpu_coeffs_l = coeffs_l.clone();
        let gpu_coeffs_r = coeffs_r.clone();
        let mut gpu_z_l = z_l.clone();
        let gpu_z_r = z_r.clone();
        let mut gpu_key_proj_l = key_proj_l.clone();
        let gpu_key_r = key_r.clone();

        coeffs_l.par_iter_mut()
            .zip(coeffs_r)
            .for_each(|(c_l, c_r)| *c_l += &(round_challenge_inv * &c_r));

        z_l.par_iter_mut()
            .zip(z_r)
            .for_each(|(z_l, z_r)| *z_l += &(round_challenge * &z_r));

        key_proj_l.par_iter_mut()
            .zip(key_r)
            .for_each(|(k_l, k_r)| *k_l += &k_r.mul(round_challenge));

        PC_DEE::polycommit_round_reduce(
            round_challenge,
            round_challenge_inv,
            &mut gpu_coeffs_l,
            &gpu_coeffs_r,
            &mut gpu_z_l,
            &gpu_z_r,
            &mut gpu_key_proj_l,
            &gpu_key_r
        );

        assert_eq!(coeffs_l, gpu_coeffs_l);
        assert_eq!(z_l, gpu_z_l);
        assert_eq!(key_proj_l, gpu_key_proj_l);
    }
}