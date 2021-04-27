use crate::{BTreeMap, String, ToString, Vec};
use crate::{Error, Evaluations, QuerySet};
use crate::{LabeledCommitment, LabeledPolynomial, LabeledRandomness};
use crate::{PCRandomness, PCUniversalParams, Polynomial, PolynomialCommitment};
use algebra::msm::VariableBaseMSM;
use algebra::{ToBytes, to_bytes, Field, PrimeField, UniformRand, Group, AffineCurve, ProjectiveCurve};
use std::{format, vec};
use std::marker::PhantomData;
use rand_core::RngCore;

mod data_structures;
pub use data_structures::*;

use rayon::prelude::*;

use digest::Digest;
use crate::rng::{FiatShamirRng, FiatShamirChaChaRng};

/// The dlog commitment scheme from Bootle et al. based on the hardness of the discrete 
/// logarithm problem in prime-order groups.
/// This implementation is according to the variant given in [[BCMS20]][pcdas], extended 
/// to support polynomials of arbitrary degree via segmentation.
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
#[derive(Derivative)]
#[derivative(Clone(bound = ""))]
pub struct InnerProductArgPC<G: AffineCurve, D: Digest> {
    _projective: PhantomData<G>,
    _digest: PhantomData<D>,
}

impl<G: AffineCurve, D: Digest> InnerProductArgPC<G, D> {
    /// `PROTOCOL_NAME` is used as a seed for the setup function.
    const PROTOCOL_NAME: &'static [u8] = b"PC-DL-2020";

    /// The low-level single segment single poly commit function.
    /// Create a dlog commitment to `scalars` using the commitment key `comm_key`.
    /// Optionally, randomize the commitment using `hiding_generator` and `randomizer`.
    pub fn cm_commit(
        comm_key: &[G],
        scalars: &[G::ScalarField],
        hiding_generator: Option<G>,
        randomizer: Option<G::ScalarField>,
    ) -> G::Projective {
        let scalars_bigint = scalars.par_iter()
            .map(|s| s.into_repr())
            .collect::<Vec<_>>();
        let mut comm = VariableBaseMSM::multi_scalar_mul(&comm_key, &scalars_bigint);
        if randomizer.is_some() {
            assert!(hiding_generator.is_some());
            comm += &hiding_generator.unwrap().mul(randomizer.unwrap());
        }
        comm
    }

    #[inline]
    fn inner_product(l: &[G::ScalarField], r: &[G::ScalarField]) -> G::ScalarField {
        l.par_iter().zip(r).map(|(li, ri)| *li * ri).sum()
    }

    /// Computes an opening proof of multiple check polynomials with a corresponding
    /// commitment GFin opened at point.
    /// No segmentation here: Bullet Polys are at most as big as the committer key.
    pub fn open_check_polys<'a>(
        ck: &CommitterKey<G>,
        xi_s: impl IntoIterator<Item = &'a SuccinctCheckPolynomial<G::ScalarField>>,
        g_fins: impl IntoIterator<Item = &'a Commitment<G>>,
        point: G::ScalarField,
        fs_rng: &mut FiatShamirChaChaRng<D>,
    ) -> Result<Proof<G>, Error>
    {
        let mut key_len = ck.comm_key.len();
        assert_eq!(ck.comm_key.len().next_power_of_two(), key_len);

        let batch_time = start_timer!(|| "Compute and batch Bullet Polys and GFin commitments");
        let xi_s_vec = xi_s.into_iter().collect::<Vec<_>>();
        let g_fins = g_fins.into_iter().collect::<Vec<_>>();

        // Compute the evaluations of the Bullet polynomials at point starting from the xi_s
        let values = xi_s_vec.par_iter().map(|xi_s| {
            xi_s.evaluate(point)
        }).collect::<Vec<_>>();

        // Absorb evaluations
        fs_rng.absorb(&values.iter().flat_map(|val| to_bytes!(val).unwrap()).collect::<Vec<_>>());

        // Sample new batching challenge
        let random_scalar: G::ScalarField = fs_rng.squeeze_128_bits_challenge();

        // Collect the powers of the batching challenge in a vector
        let mut batching_chal = G::ScalarField::one();
        let mut batching_chals = vec![G::ScalarField::zero(); xi_s_vec.len()];
        for i in 0..batching_chals.len() {
            batching_chals[i] = batching_chal;
            batching_chal *= &random_scalar;
        }

        // Compute combined check_poly and combined g_fin
        let (mut combined_check_poly, combined_g_fin, combined_v) = batching_chals
            .into_par_iter()
            .zip(xi_s_vec)
            .zip(values)
            .zip(g_fins)
            .map(|(((chal, xi_s), value), g_fin)| {
                (
                    Polynomial::from_coefficients_vec(xi_s.compute_scaled_coeffs(chal)),
                    g_fin.comm[0].mul(chal),
                    chal * value
                )
            }).reduce(
                || (Polynomial::zero(), <G::Projective as ProjectiveCurve>::zero(), G::ScalarField::zero()),
                |acc, poly_comm_val| (&acc.0 + &poly_comm_val.0, acc.1 + &poly_comm_val.1, acc.2 + &poly_comm_val.2)
            );

        // It's not necessary to use the full length of the ck if all the Bullet Polys are smaller:
        // trim the ck if that's the case
        key_len = combined_check_poly.coeffs.len();
        assert_eq!(key_len.next_power_of_two(), key_len);
        let mut comm_key = &ck.comm_key[..key_len];

        end_timer!(batch_time);

        let proof_time =
            start_timer!(|| format!("Generating proof for degree {} combined polynomial", key_len));

        let combined_g_fin = combined_g_fin.into_affine();

        // ith challenge
        fs_rng.absorb(&to_bytes![combined_g_fin, point, combined_v].unwrap());
        let mut round_challenge: G::ScalarField = fs_rng.squeeze_128_bits_challenge();

        let h_prime = ck.h.mul(round_challenge).into_affine();

        let mut coeffs = combined_check_poly.coeffs.as_mut_slice();

        // Powers of z
        let mut z: Vec<G::ScalarField> = Vec::with_capacity(key_len);
        let mut cur_z: G::ScalarField = G::ScalarField::one();
        for _ in 0..key_len {
            z.push(cur_z);
            cur_z *= &point;
        }
        let mut z = z.as_mut_slice();

        // This will be used for transforming the key in each step
        let mut key_proj: Vec<G::Projective> = comm_key.iter().map(|x| (*x).into_projective()).collect();
        let mut key_proj = key_proj.as_mut_slice();

        let mut temp;

        let log_key_len = algebra::log2(key_len) as usize;
        let mut l_vec = Vec::with_capacity(log_key_len);
        let mut r_vec = Vec::with_capacity(log_key_len);

        let mut n = key_len;
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

            fs_rng.absorb(&to_bytes![lr[0], lr[1]].unwrap());
            round_challenge = fs_rng.squeeze_128_bits_challenge();

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
            hiding_comm: None,
            rand: None,
        })
    }

    /// The succinct portion of verifying a multi-poly single-point opening proof.
    /// If successful, returns the (recomputed) reduction challenge.
    pub fn succinct_check<'a>(
        vk: &VerifierKey<G>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Commitment<G>>>,
        point: G::ScalarField,
        values: impl IntoIterator<Item = G::ScalarField>,
        proof: &Proof<G>,
        // This implementation assumes that the commitments, point and evaluations are 
        // already bound to the internal state of the Fiat Shamir rng
        fs_rng: &mut FiatShamirChaChaRng<D>,
    ) -> Result<Option<SuccinctCheckPolynomial<G::ScalarField>>, Error> {
        let check_time = start_timer!(|| "Succinct checking");

        // We do not assume that the vk length is equal to the segment size.
        // Instead, we read the segment size from the proof L and Rs vectors (i.e. the number
        // of steps of the dlog reduction). Doing so allows us to easily verify 
        // the dlog opening proofs produced by different size-restricted by means
        // of a single vk.
        let log_key_len = proof.l_vec.len();
        let key_len = 1 << log_key_len;

        if proof.l_vec.len() != proof.r_vec.len() || proof.l_vec.len() > log_key_len {
            return Err(Error::IncorrectInputLength(
                format!(
                    "Expected proof vectors to be at most {:}. Instead, l_vec size is {:} and r_vec size is {:}",
                    log_key_len,
                    proof.l_vec.len(),
                    proof.r_vec.len()
                )
            ));
        }

        let mut combined_commitment_proj = <G::Projective as ProjectiveCurve>::zero();
        let mut combined_v = G::ScalarField::zero();

        let lambda: G::ScalarField = fs_rng.squeeze_128_bits_challenge();
        let mut cur_challenge = G::ScalarField::one();

        let labeled_commitments = commitments.into_iter();
        let values = values.into_iter();

        for (labeled_commitment, value) in labeled_commitments.zip(values) {
            let label = labeled_commitment.label();
            let commitment = labeled_commitment.commitment();
            combined_v += &(cur_challenge * &value);

            let segments_count = commitment.comm.len();

            let mut comm_lc = <G::Projective as ProjectiveCurve>::zero();
            for (i, comm_single) in commitment.comm.iter().enumerate() {
                let is = i * key_len;
                comm_lc += &comm_single.mul(point.pow(&[is as u64]));
            }
            combined_commitment_proj += &comm_lc.mul(&cur_challenge);

            cur_challenge = cur_challenge * &lambda;

            let degree_bound = labeled_commitment.degree_bound();

            // If the degree_bound is a multiple of the key_len then there is no need to prove the degree bound polynomial identity.
            let degree_bound_len = degree_bound.and_then(|degree_bound_len| {
                if (degree_bound_len + 1) % key_len != 0 { Some(degree_bound_len + 1) } else { None }
            });

            assert_eq!(
                degree_bound_len.is_some(),
                commitment.shifted_comm.is_some()
            );

            if let Some(degree_bound_len) = degree_bound_len {

                if Self::check_segments_and_bounds(
                    degree_bound.unwrap(),
                    segments_count,
                    key_len,
                    label.clone(),
                ).is_err() {
                    return Ok(None);
                }

                let shifted_degree_bound = degree_bound_len % key_len - 1;
                let shift = -point.pow(&[(key_len - shifted_degree_bound - 1) as u64]);
                combined_commitment_proj += &commitment.shifted_comm.unwrap().mul(cur_challenge);
                combined_commitment_proj += &commitment.comm[segments_count - 1].mul(cur_challenge * &shift);

                cur_challenge = cur_challenge * &lambda;
            }
        }

        assert_eq!(proof.hiding_comm.is_some(), proof.rand.is_some());
        if proof.hiding_comm.is_some() {
            let hiding_comm = proof.hiding_comm.unwrap();
            let rand = proof.rand.unwrap();

            fs_rng.absorb(&to_bytes![hiding_comm].unwrap());
            let hiding_challenge: G::ScalarField = fs_rng.squeeze_128_bits_challenge();
            fs_rng.absorb(&(to_bytes![rand].unwrap()));

            combined_commitment_proj += &(hiding_comm.mul(hiding_challenge) - &vk.s.mul(rand));
        }

        // Challenge for each round
        let mut round_challenges = Vec::with_capacity(log_key_len);

        let mut round_challenge: G::ScalarField = fs_rng.squeeze_128_bits_challenge();

        let h_prime = vk.h.mul(round_challenge);

        let mut round_commitment_proj = combined_commitment_proj + &h_prime.mul(&combined_v);

        let l_iter = proof.l_vec.iter();
        let r_iter = proof.r_vec.iter();

        for (l, r) in l_iter.zip(r_iter) {

            fs_rng.absorb(&to_bytes![l, r].unwrap());
            round_challenge = fs_rng.squeeze_128_bits_challenge();

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
            return Ok(None)
        }

        end_timer!(check_time);
        Ok(Some(check_poly))
    }

    /// Succinct check of a multi-point multi-poly opening proof from [[BDFG2020]](https://eprint.iacr.org/2020/081) 
    /// If successful, returns the (recomputed) succinct check polynomial (the xi_s) 
    /// and the GFinal.
    pub fn succinct_batch_check_individual_opening_challenges<'a>(
        vk: &VerifierKey<G>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Commitment<G>>>,
        query_set: &QuerySet<G::ScalarField>,
        values: &Evaluations<G::ScalarField>,
        batch_proof: &BatchProof<G>,
        // This implementation assumes that the commitments, query set and evaluations are already absorbed by the Fiat Shamir rng
        fs_rng: &mut FiatShamirChaChaRng<D>,
    ) -> Result<(SuccinctCheckPolynomial<G::ScalarField>, G), Error>
    {
        let commitments: Vec<&'a LabeledCommitment<Commitment<G>>> = commitments.into_iter().map(|comm| comm).collect();

        let batch_check_time = start_timer!(|| "Multi poly multi point batch check: succinct part");
        let evals_time = start_timer!(|| "Compute batched poly value");

        // v_i values
        let mut v_values = vec![];

        // y_i values
        let mut y_values = vec![];

        // x_i values
        let mut points = vec![];

        for (label, (_point_label, point)) in query_set.iter() {

            let y_i = values
                .get(&(label.clone(), *point))
                .ok_or(Error::MissingEvaluation {
                    label: label.to_string(),
                })?;
            let v_i = batch_proof.batch_values
                .get(&label.clone())
                .ok_or(Error::MissingEvaluation {
                    label: label.to_string(),
                })?;
            v_values.push(*v_i);
            y_values.push(*y_i);
            points.push(point);
        }

        // Commitment of the h(X) polynomial
        let batch_commitment = batch_proof.batch_commitment.clone();

        // lambda
        let lambda: G::ScalarField = fs_rng.squeeze_128_bits_challenge();
        let mut cur_challenge = G::ScalarField::one();

        // Fresh random challenge x
        fs_rng.absorb(&to_bytes![batch_commitment].unwrap());
        let point: G::ScalarField = fs_rng.squeeze_128_bits_challenge();

        let mut computed_batch_v = G::ScalarField::zero();

        for ((&v_i, y_i), x_i) in v_values.iter().zip(y_values).zip(points) {

            computed_batch_v = computed_batch_v + &(cur_challenge * &((v_i - &y_i) / &(point - x_i)));

            cur_challenge = cur_challenge * &lambda;
        }

        let mut batch_values = vec![];
        for commitment in commitments.iter() {
            let value = batch_proof.batch_values
                .get(commitment.label())
                .ok_or(Error::MissingEvaluation {
                    label: commitment.label().to_string(),
                })?;
            batch_values.push(*value);
        }

        // Reconstructed v value added to the check
        batch_values.push(computed_batch_v);

        // The commitment to h(X) polynomial added to the check
        let mut commitments = commitments;
        let labeled_batch_commitment = LabeledCommitment::new(
            format!("Batch"),
            Commitment { comm: batch_commitment.clone(), shifted_comm: None },
            None
        );
        commitments.push(&labeled_batch_commitment);

        let proof = &batch_proof.proof;
        end_timer!(evals_time);

        let check_time = start_timer!(|| "Succinct check batched polynomial");

        fs_rng.absorb(&to_bytes![batch_proof.batch_values.values().collect::<Vec<&G::ScalarField>>()].unwrap());

        let check_poly = Self::succinct_check(vk, commitments, point, batch_values, proof, fs_rng)?;

        if check_poly.is_none() {
            end_timer!(check_time);
            end_timer!(batch_check_time);
            return Err(Error::FailedSuccinctCheck);
        }

        end_timer!(check_time);
        end_timer!(batch_check_time);

        Ok((check_poly.unwrap(), proof.final_comm_key))
    }

    /// Succinct verify (a batch of) multi-point mulit-poly opening proofs and, if valid, 
    /// return their SuccinctCheckPolynomials (the reduction challenges `xi`) and the
    /// final committer keys `GFinal`.
    pub fn succinct_batch_check<'a>(
        vk:                 &VerifierKey<G>,
        commitments:        impl IntoIterator<Item = &'a [LabeledCommitment<Commitment<G>>]>,
        query_sets:         impl IntoIterator<Item = &'a QuerySet<'a, G::ScalarField>>,
        values:             impl IntoIterator<Item = &'a Evaluations<'a, G::ScalarField>>,
        proofs:             impl IntoIterator<Item = &'a BatchProof<G>>,
        states:             impl IntoIterator<Item = &'a <FiatShamirChaChaRng<D> as FiatShamirRng>::Seed>,
    ) -> Result<(Vec<SuccinctCheckPolynomial<G::ScalarField>>, Vec<G>), Error>
        where
            D::OutputSize: 'a
    {
        let comms = commitments.into_iter().collect::<Vec<_>>();
        let query_sets = query_sets.into_iter().collect::<Vec<_>>();
        let values = values.into_iter().collect::<Vec<_>>();
        let proofs = proofs.into_iter().collect::<Vec<_>>();
        let states = states.into_iter().collect::<Vec<_>>();

        // Perform succinct verification of all the proofs and collect
        // the xi_s and the GFinal_s into DLogAccumulators
        let succinct_time = start_timer!(|| "Succinct verification of proofs");

        let accumulators = comms.into_par_iter()
            .zip(query_sets)
            .zip(values)
            .zip(proofs)
            .zip(states)
            .map(|((((commitments, query_set), values), proof), state)|
                {
                    let mut fs_rng = FiatShamirChaChaRng::<D>::new();
                    fs_rng.set_seed(state.clone());

                    // Perform succinct check of i-th proof
                    let (challenges, final_comm_key) = Self::succinct_batch_check_individual_opening_challenges(
                        vk,
                        commitments,
                        query_set,
                        values,
                        proof,
                        &mut fs_rng,
                    ).unwrap();

                    (final_comm_key, challenges)
                }
            ).collect::<Vec<_>>();
        end_timer!(succinct_time);

        let g_finals = accumulators.iter().map(|(g_final, _)| g_final.clone()).collect::<Vec<_>>();
        let challenges = accumulators.into_iter().map(|(_, xi_s)| xi_s).collect::<Vec<_>>();

        Ok((challenges, g_finals))
    }

    /// Checks whether degree bounds are `situated' in the last segment of a polynomial
    /// TODO: Rename to check_bounds, or alternatively write a function that receives the
    ///       supposed degree, and which checks in addition whether the segment count is plausible.
    fn check_degrees_and_bounds(
        supported_degree: usize,
        p: &LabeledPolynomial<G::ScalarField>,
    ) -> Result<(), Error> {
        // We use segmentation, therefore we allow arbitrary degree polynomials: hence, the only
        // check that makes sense, is the bound being bigger than the degree of the polynomial.
        if let Some(bound) = p.degree_bound() {

            let p_len = p.polynomial().coeffs.len();
            let segment_len = supported_degree + 1;
            let segments_count = std::cmp::max(1, p_len / segment_len + if p_len % segment_len != 0 { 1 } else { 0 });

            if bound < p.degree() {
                return Err(Error::IncorrectDegreeBound {
                    poly_degree: p.degree(),
                    degree_bound: bound,
                    supported_degree,
                    label: p.label().to_string(),
                });
            }

            return Self::check_segments_and_bounds(
                bound,
                segments_count,
                segment_len,
                p.label().to_string()
            );
        }

        Ok(())
    }

    /// Checks if the degree bound is situated in the last segment.
    fn check_segments_and_bounds(
        bound: usize,
        segments_count: usize,
        segment_len: usize,
        label: String
    ) -> Result<(), Error> {

        if (bound + 1) <= (segments_count-1) * segment_len ||
            (bound + 1) > segments_count * segment_len
        {
            return Err(Error::IncorrectSegmentedDegreeBound {
                degree_bound: bound,
                segments_count,
                segment_len,
                label,
            });
        }

        Ok(())
    }

    /// Computes the 'shifted' polynomial as needed for degree bound proofs.
    fn shift_polynomial(
        ck: &CommitterKey<G>,
        p: &Polynomial<G::ScalarField>,
        degree_bound: usize,
    ) -> Polynomial<G::ScalarField> {
        if p.is_zero() {
            Polynomial::zero()
        } else {
            let mut shifted_polynomial_coeffs =
                vec![G::ScalarField::zero(); ck.comm_key.len() - 1 - degree_bound];
            shifted_polynomial_coeffs.extend_from_slice(&p.coeffs);
            Polynomial::from_coefficients_vec(shifted_polynomial_coeffs)
        }
    }

    /// Computing the base point vector of the commmitment scheme in a 
    /// deterministic manner, given the PROTOCOL_NAME.
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

    /// Perform a dlog reduction step as described in BCMS20
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

/// Implementation of the PolynomialCommitment trait for the segmentized dlog commitment scheme 
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
    type RandomOracle = FiatShamirChaChaRng<D>;

    /// Setup of the base point vector (deterministically derived from the
    /// PROTOCOL_NAME as seed).
    fn setup(
        max_degree: usize,
    ) -> Result<Self::UniversalParams, Self::Error> {
        // Ensure that max_degree + 1 is a power of 2
        let max_degree = (max_degree + 1).next_power_of_two() - 1;

        let setup_time = start_timer!(|| format!("Sampling {} generators", max_degree + 3));
        let mut generators = Self::sample_generators(max_degree + 3);
        end_timer!(setup_time);

        let hash = D::digest(&to_bytes![generators, max_degree as u32].unwrap()).to_vec();

        let h = generators.pop().unwrap();
        let s = generators.pop().unwrap();

        let pp = UniversalParams {
            comm_key: generators,
            h,
            s,
            hash,
        };

        Ok(pp)
    }

    /// Trims the base point vector of the setup function to a custom segment size
    fn trim(
        pp: &Self::UniversalParams,
        // the segment size (TODO: let's rename it!)
        supported_degree: usize,
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
            hash: pp.hash.clone(),
        };

        let vk = VerifierKey {
            comm_key: pp.comm_key[0..(supported_degree + 1)].to_vec(),
            h: pp.h.clone(),
            s: pp.s.clone(),
            max_degree: pp.max_degree(),
            hash: pp.hash.clone(),
        };

        end_timer!(trim_time);

        Ok((ck, vk))
    }

    /// Domain extended commit function, outputs a `segmented commitment' 
    /// to a polynomial, regardless of its degree.
    fn commit<'a>(
        ck: &Self::CommitterKey,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField>>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<
        (
            Vec<LabeledCommitment<Self::Commitment>>,
            Vec<LabeledRandomness<Self::Randomness>>,
        ),
        Self::Error,
    > {
        let rng = &mut crate::optional_rng::OptionalRng(rng);
        let mut comms = Vec::new();
        let mut rands = Vec::new();

        let commit_time = start_timer!(|| "Committing to polynomials");
        for labeled_polynomial in polynomials {
            Self::check_degrees_and_bounds(ck.comm_key.len() - 1, labeled_polynomial)?;

            let polynomial = labeled_polynomial.polynomial();
            let label = labeled_polynomial.label();
            let hiding_bound = labeled_polynomial.hiding_bound();
            let degree_bound = labeled_polynomial.degree_bound();

            let single_commit_time = start_timer!(|| format!(
                "Polynomial {} of degree {}, degree bound {:?}, and hiding bound {:?}",
                label,
                polynomial.degree(),
                degree_bound,
                hiding_bound,
            ));

            let key_len = ck.comm_key.len();
            let p_len = polynomial.coeffs.len();
            let segments_count = std::cmp::max(1, p_len / key_len + if p_len % key_len != 0 { 1 } else { 0 });

            let randomness = if let Some(_) = hiding_bound {
                Randomness::rand(segments_count, degree_bound.is_some(), rng)
            } else {
                Randomness::empty(segments_count)
            };

            let comm: Vec<G>;

            // split poly in segments and commit all of them without shifting
            comm = (0..segments_count).into_iter().map(
                |i| {
                    Self::cm_commit(
                        &ck.comm_key,
                        &polynomial.coeffs[i * key_len..core::cmp::min((i + 1) * key_len, p_len)],
                        Some(ck.s),
                        Some(randomness.rand[i]),
                    ).into_affine()
                }
            ).collect();

            // committing only last segment shifted to the right edge
            let shifted_comm = degree_bound.and_then(|degree_bound| {
                let degree_bound_len = degree_bound + 1; // Convert to the maximum number of coefficients
                if degree_bound_len % key_len != 0 {
                    Some(
                        Self::cm_commit(
                            &ck.comm_key[key_len - (degree_bound_len % key_len)..],
                            &polynomial.coeffs[(segments_count - 1) * key_len..p_len],
                            Some(ck.s),
                            randomness.shifted_rand,
                        ).into_affine()
                    )
                } else {
                    None
                }
            });

            let commitment = Commitment { comm, shifted_comm };
            let labeled_comm = LabeledCommitment::new(label.to_string(), commitment, degree_bound);
            let labeled_rand = LabeledRandomness::new(label.to_string(), randomness);

            comms.push(labeled_comm);
            rands.push(labeled_rand);

            end_timer!(single_commit_time);
        }

        end_timer!(commit_time);
        Ok((comms, rands))
    }

    /// Single point multi poly open, allowing the random oracle to be passed from 
    /// 'outside' to the function. 
    /// CAUTION: This is a low-level function which assumes that the statement of the
    /// opening proof (i.e. commitments, query point, and evaluations) is already bound 
    /// to the internal state of the Fiat-Shamir rng.
    fn open_individual_opening_challenges<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: G::ScalarField,
        // This implementation assumes that commitments, query point and evaluations are already absorbed by the Fiat Shamir rng
        fs_rng: &mut Self::RandomOracle,
        rands: impl IntoIterator<Item = &'a LabeledRandomness<Self::Randomness>>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Self::Error>
        where
            Self::Commitment: 'a,
            Self::Randomness: 'a,
    {
        let key_len = ck.comm_key.len();
        let log_key_len = algebra::log2(key_len) as usize;

        assert_eq!(key_len.next_power_of_two(), key_len);

        let mut combined_polynomial = Polynomial::zero();
        let mut combined_rand = G::ScalarField::zero();
        let mut combined_commitment_proj = <G::Projective as ProjectiveCurve>::zero();

        let mut has_hiding = false;

        let polys_iter = labeled_polynomials.into_iter();
        let rands_iter = rands.into_iter();
        let comms_iter = commitments.into_iter();

        let combine_time = start_timer!(|| "Combining polynomials, randomness, and commitments.");

        // as the statement of the opening proof is already bound to the interal state of the fr_rng,
        // we simply squeeze the challenge scalar for the random linear combination
        let lambda: G::ScalarField = fs_rng.squeeze_128_bits_challenge();
        let mut cur_challenge = G::ScalarField::one();

        // compute the random linear combination using the powers of lambda
        for (labeled_polynomial, (labeled_commitment, labeled_randomness)) in
        polys_iter.zip(comms_iter.zip(rands_iter))
        {
            let label = labeled_polynomial.label();
            assert_eq!(labeled_polynomial.label(), labeled_commitment.label());
            Self::check_degrees_and_bounds(ck.comm_key.len() - 1, labeled_polynomial)?;

            let polynomial = labeled_polynomial.polynomial();
            let degree_bound = labeled_polynomial.degree_bound();
            let hiding_bound = labeled_polynomial.hiding_bound();
            let commitment = labeled_commitment.commitment();
            let randomness = labeled_randomness.randomness();

            let p_len = polynomial.coeffs.len();
            let segments_count = std::cmp::max(1, p_len / key_len + if p_len % key_len != 0 { 1 } else { 0 });

            // If the degree_bound is a multiple of the key_len then there is no need to prove the degree bound polynomial identity.
            let degree_bound_len = degree_bound.and_then(|degree_bound| {
                if (degree_bound + 1) % key_len != 0 { Some(degree_bound + 1) } else { None }
            });

            assert_eq!(
                degree_bound_len.is_some(),
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

            if hiding_bound.is_some() {
                has_hiding = true;
            }

            let mut polynomial_lc = Polynomial::zero();
            let mut comm_lc = <G::Projective as ProjectiveCurve>::zero();
            let mut rand_lc = G::ScalarField::zero();

            // Compute the query-point dependent linear combination of the segments,
            // both for witnesses, commitments (and their randomnesses, if hiding)
            for i in 0..segments_count {
                let is = i * key_len;
                let poly_single = Polynomial::from_coefficients_slice(
                    &polynomial.coeffs[i * key_len..core::cmp::min((i + 1) * key_len, p_len)]
                );
                let comm_single = commitment.comm[i];
                // add x^{i*|S|}* p_i(X) of the segment polynomial p_i(X)
                polynomial_lc += (point.pow(&[is as u64]), &poly_single);
                comm_lc += &comm_single.mul(point.pow(&[is as u64]));
                if has_hiding {
                    let rand_single = randomness.rand[i];
                    rand_lc += &(point.pow(&[is as u64]) * rand_single);
                }
            }

            // add segment linear combination to overall combination, 
            // both for witnesses, commitments (and their randomnesses, if hiding)
            combined_polynomial += (cur_challenge, &polynomial_lc);
            combined_commitment_proj += &comm_lc.mul(&cur_challenge);
            if has_hiding {
                combined_rand += &(cur_challenge * &rand_lc);
            }

            // next power of lambda
            cur_challenge = cur_challenge * &lambda;

            // If we prove degree bound, we add the degree bound identity 
            //  p_shift(X) - x^{|S|-d} p(X),
            // where p(X) is the last segment polynomial, d its degree, |S| the
            // segment size and p_shift(X) the shifted polynomial.
            if let Some(degree_bound_len) = degree_bound_len {

                // degree bound relative to the last segment
                let shifted_degree_bound = degree_bound_len % key_len - 1;
                let last_segment_polynomial = Polynomial::from_coefficients_slice(
                    &polynomial.coeffs[(segments_count - 1) * key_len..p_len]
                );
                let shifted_polynomial = Self::shift_polynomial(
                    ck,
                    &last_segment_polynomial,
                    shifted_degree_bound
                );
                let shift = -point.pow(&[(key_len - shifted_degree_bound - 1) as u64]);

                // add the shifted polynomial p_shift(X) and its commitment
                combined_polynomial += (cur_challenge, &shifted_polynomial);
                combined_commitment_proj += &commitment.shifted_comm.unwrap().mul(cur_challenge);

                // add -x^{N-d} * p(X) and its commitment
                combined_polynomial += (cur_challenge * &shift, &last_segment_polynomial);
                combined_commitment_proj += &commitment.comm[segments_count - 1].mul(cur_challenge * &shift);

                // add the randomnesses accordingly
                if hiding_bound.is_some() {
                    let shifted_rand = randomness.shifted_rand;
                    assert!(
                        shifted_rand.is_some(),
                        "shifted_rand.is_none() for {}",
                        label
                    );
                    // randomness of p_shift(X)
                    combined_rand += &(cur_challenge * &shifted_rand.unwrap());
                    // randomness of -x^{N-d} * p(X)
                    combined_rand += &(cur_challenge * &shift * &randomness.rand[segments_count - 1]);
                }

                // next power of lamba
                cur_challenge = cur_challenge * &lambda;
            }
        }

        end_timer!(combine_time);

        let mut hiding_commitment = None;

        if has_hiding {
            let mut rng = rng.expect("hiding commitments require randomness");
            let hiding_time = start_timer!(|| "Applying hiding.");
            let mut hiding_polynomial = Polynomial::rand(key_len - 1, &mut rng);
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

            // We assume that the commitments, the query point, and the evaluations are already
            // bound to the internal state of the Fiat-Shamir rng. Hence the same is true for 
            // the deterministically derived combined_commitment and its combined_v.
            fs_rng.absorb(&to_bytes![hiding_commitment.unwrap()].unwrap());
            let hiding_challenge: G::ScalarField = fs_rng.squeeze_128_bits_challenge();

            // compute random linear combination using the hiding_challenge, 
            // both for witnesses and commitments (and it's randomness)
            combined_polynomial += (hiding_challenge, &hiding_polynomial);
            combined_rand += &(hiding_challenge * &hiding_rand);
            fs_rng.absorb(&to_bytes![combined_rand].unwrap());
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
            start_timer!(|| format!("Generating proof for degree {} combined polynomial", key_len));

        // 0-th challenge
        let mut round_challenge: G::ScalarField = fs_rng.squeeze_128_bits_challenge();

        let h_prime = ck.h.mul(round_challenge).into_affine();

        // Pads the coefficients with zeroes to get the number of coeff to be key_len
        let mut coeffs = combined_polynomial.coeffs;
        if coeffs.len() < key_len {
            for _ in coeffs.len()..key_len {
                coeffs.push(G::ScalarField::zero());
            }
        }
        let mut coeffs = coeffs.as_mut_slice();

        // Powers of z
        let mut z: Vec<G::ScalarField> = Vec::with_capacity(key_len);
        let mut cur_z: G::ScalarField = G::ScalarField::one();
        for _ in 0..key_len {
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

        let mut l_vec = Vec::with_capacity(log_key_len);
        let mut r_vec = Vec::with_capacity(log_key_len);

        let mut n = key_len;
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

            // the previous challenge is bound to the internal state, hence 
            // no need to absorb it
            fs_rng.absorb(&to_bytes![lr[0], lr[1]].unwrap());

            round_challenge = fs_rng.squeeze_128_bits_challenge();
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

    /// The multi point multi poly opening proof from [[BDFG2020]](https://eprint.iacr.org/2020/081) 
    /// CAUTION: This is a low-level function which assumes that the statement of the
    /// opening proof (i.e. commitments, query point, and evaluations) is already bound 
    /// to the internal state of the Fiat-Shamir rng.
    fn batch_open_individual_opening_challenges<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<G::ScalarField>,
        // This implementation assumes that the commitments (as well as the query set and evaluations)
        // are already absorbed by the Fiat Shamir rng
        fs_rng: &mut Self::RandomOracle,
        rands: impl IntoIterator<Item = &'a LabeledRandomness<Self::Randomness>>,
        mut rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::BatchProof, Self::Error>
        where
            Self::Randomness: 'a,
            Self::Commitment: 'a,
    {
        let labeled_polynomials: Vec<&'a LabeledPolynomial<G::ScalarField>> = labeled_polynomials.into_iter().map(|poly| poly).collect();
        let commitments: Vec<&'a LabeledCommitment<Self::Commitment>> = commitments.into_iter().map(|comm| comm).collect();
        let rands: Vec<&'a LabeledRandomness<Self::Randomness>> = rands.into_iter().map(|rand| rand).collect();

        let batch_time = start_timer!(|| "Multi poly multi point batching.");

        // as the statement of the opening proof is already bound to the interal state of the fs_rng,
        // we simply squeeze the challenge scalar for the random linear combination
        let lambda: G::ScalarField = fs_rng.squeeze_128_bits_challenge();
        let mut cur_challenge = G::ScalarField::one();

        let poly_map: BTreeMap<_, _> = labeled_polynomials
            .iter()
            .map(|poly| (poly.label(), poly))
            .collect();

        let mut has_hiding = false;
        let mut points = vec![];

        // h(X)
        let h_poly_time = start_timer!(|| "Compute batching polynomial");
        let mut batch_polynomial = Polynomial::zero();

        for (label, (_point_label, point)) in query_set.iter() {
            let labeled_polynomial =
                poly_map.get(label).ok_or(Error::MissingPolynomial {
                    label: label.to_string(),
                })?;

            if labeled_polynomial.hiding_bound().is_some() {
                has_hiding = true;
            }

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
            cur_challenge = cur_challenge * &lambda;
        }
        end_timer!(h_poly_time);

        let key_len = ck.comm_key.len();
        let p_len = batch_polynomial.coeffs.len();
        let segments_count = std::cmp::max(1, p_len / key_len + if p_len % key_len != 0 { 1 } else { 0 });

        let batch_randomness = if has_hiding {
            Randomness::rand(segments_count, false, rng.as_mut().unwrap())
        } else {
            Randomness::empty(segments_count)
        };

        // Commitment of the h(X) polynomial
        let commit_time = start_timer!(|| format!("Commit to batch polynomial of degree {}", batch_polynomial.degree()));
        let batch_commitment: Vec<G>;

        if p_len > key_len {

            batch_commitment = (0..segments_count).into_iter().map(
                |i| {
                    Self::cm_commit(
                        &ck.comm_key,
                        &batch_polynomial.coeffs[i * key_len..core::cmp::min((i + 1) * key_len, p_len)],
                        Some(ck.s),
                        Some(batch_randomness.rand[i]),
                    ).into_affine()
                }
            ).collect();

        } else {

            batch_commitment = vec![
                Self::cm_commit(
                    ck.comm_key.as_slice(),
                    batch_polynomial.coeffs.as_slice(),
                    Some(ck.s),
                    Some(batch_randomness.rand[0]),
                ).into_affine()
            ];
        }
        end_timer!(commit_time);

        let open_time = start_timer!(|| "Open batch polynomial");

        // Fresh random challenge x for multi-point to single-point reduction.
        // Except the `batch_commitment`, all other commitments are already bound 
        // to the internal state of the Fiat-Shamir
        fs_rng.absorb(&to_bytes![batch_commitment].unwrap());
        let point: G::ScalarField = fs_rng.squeeze_128_bits_challenge();

        // Values: v_i = p_i(x), where x is fresh random challenge
        let batch_values: BTreeMap<_, _> = labeled_polynomials
            .iter()
            .map(|labeled_polynomial| (labeled_polynomial.label().clone(), labeled_polynomial.polynomial().evaluate(point)))
            .collect();

        // h(X) polynomial added to the set of polynomials for multi-poly single-point batching
        let mut labeled_polynomials = labeled_polynomials;
        let labeled_batch_polynomial = LabeledPolynomial::new(
            format!("Batch"),
            batch_polynomial,
            None,
            if has_hiding { Some(1) } else { None }
        );
        labeled_polynomials.push(&labeled_batch_polynomial);

        // Commitment of h(X) polynomial added to the set of polynomials for multi-poly single-point batching
        let mut commitments = commitments;
        let labeled_batch_commitment = LabeledCommitment::new(
            format!("Batch"),
            Commitment { comm: batch_commitment.clone(), shifted_comm: None },
            None
        );
        commitments.push(&labeled_batch_commitment);

        let mut rands = rands;
        let labeled_batch_rand = LabeledRandomness::new(format!("Batch"), batch_randomness);
        rands.push(&labeled_batch_rand);

        // absorb the evaluations at the new challenge x
        // The value of `batch_commitment` is determined by these and the initial 
        // opening claims
        fs_rng.absorb(&to_bytes![batch_values.values().collect::<Vec<&G::ScalarField>>()].unwrap());

        let proof = Self::open_individual_opening_challenges(
            ck,
            labeled_polynomials,
            commitments,
            point,
            fs_rng,
            rands,
            rng,
        )?;
        end_timer!(open_time);

        end_timer!(batch_time);

        Ok(BatchProof {
            proof,
            batch_commitment,
            batch_values
        })
    }


    /// The verification function of an opening proof produced by ``open_individual_opening_challenges()``
    fn check_individual_opening_challenges<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: G::ScalarField,
        values: impl IntoIterator<Item = G::ScalarField>,
        proof: &Self::Proof,
        // This implementation assumes that the commitments, point and evaluations are already absorbed by the Fiat Shamir rng
        fs_rng: &mut Self::RandomOracle,
    ) -> Result<bool, Self::Error>
        where
            Self::Commitment: 'a,
    {
        let check_time = start_timer!(|| "Checking evaluations");

        let check_poly =
            Self::succinct_check(vk, commitments, point, values, proof, fs_rng)?;

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

    /// Verifies a multi-point multi-poly opening proof from [[BDFG2020]](https://eprint.iacr.org/2020/081).
    fn batch_check_individual_opening_challenges<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<G::ScalarField>,
        evaluations: &Evaluations<G::ScalarField>,
        batch_proof: &Self::BatchProof,
        // This implementation assumes that commitments, query set and evaluations are already absorbed by the Fiat Shamir rng
        fs_rng: &mut Self::RandomOracle,
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
            fs_rng,
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
}

#[cfg(test)]
mod tests {
    #![allow(non_camel_case_types)]

    use super::InnerProductArgPC;

    use algebra::curves::tweedle::dee::{
        Affine, Projective,
    };
    use blake2::Blake2s;

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
    fn two_poly_four_points_test() {
        use crate::tests::*;
        two_poly_four_points_test::<_, PC_DEE>()
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
    fn segmented_test() {
        use crate::tests::*;
        segmented_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
        println!("Finished tweedle_dee-blake2s");
    }

    // #[test]
    // fn single_equation_test() {
    //     use crate::tests::*;
    //     single_equation_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
    //     println!("Finished tweedle_dee-blake2s");
    // }
    //
    // #[test]
    // fn two_equation_test() {
    //     use crate::tests::*;
    //     two_equation_test::<_, PC_DEE>().expect("test failed for tweedle_dee-blake2s");
    //     println!("Finished tweedle_dee-blake2s");
    // }
    //
    // #[test]
    // fn two_equation_degree_bound_test() {
    //     use crate::tests::*;
    //     two_equation_degree_bound_test::<_, PC_DEE>()
    //         .expect("test failed for tweedle_dee-blake2s");
    //     println!("Finished tweedle_dee-blake2s");
    // }
    //
    // #[test]
    // fn full_end_to_end_equation_test() {
    //     use crate::tests::*;
    //     full_end_to_end_equation_test::<_, PC_DEE>()
    //         .expect("test failed for tweedle_dee-blake2s");
    //     println!("Finished tweedle_dee-blake2s");
    // }

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