use crate::*;
use crate::{PCCommitterKey, PCVerifierKey, Vec};
use algebra::{
    Field, UniformRand, AffineCurve, PrimeField,
};
use std::{
    io::{ Read, Write }, vec, convert::TryFrom,
};
use rand_core::RngCore;

/// `UniversalParams` are the universal parameters for the inner product arg scheme.
#[derive(Derivative)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""))]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniversalParams<G: AffineCurve> {
    /// The key used to commit to polynomials.
    pub comm_key: Vec<G>,

    /// Some group generator.
    pub h: G,

    /// Some group generator specifically used for hiding.
    pub s: G,

    /// The hash of the previous fields
    pub hash: Vec<u8>,
}

impl<G: AffineCurve> PCUniversalParams for UniversalParams<G> {
    fn max_degree(&self) -> usize {
        self.comm_key.len() - 1
    }
}

/// `CommitterKey` is used to commit to, and create evaluation proofs for, a given
/// polynomial.
#[derive(Derivative)]
#[derivative(
Default(bound = ""),
Hash(bound = ""),
Clone(bound = ""),
Debug(bound = ""),
Eq(bound = ""),
PartialEq(bound = ""),
)]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct CommitterKey<G: AffineCurve> {
    /// The key used to commit to polynomials.
    pub comm_key: Vec<G>,

    /// A random group generator.
    pub h: G,

    /// A random group generator that is to be used to make
    /// a commitment hiding.
    pub s: G,

    /// The maximum degree supported by the parameters
    /// this key was derived from.
    pub max_degree: usize,

    /// The hash of all the previous fields
    pub hash: Vec<u8>,
}

impl<G: AffineCurve> PCCommitterKey for CommitterKey<G> {
    fn max_degree(&self) -> usize {
        self.max_degree
    }
    fn supported_degree(&self) -> usize {
        self.comm_key.len() - 1
    }
}

/// `VerifierKey` is used to check evaluation proofs for a given commitment.
pub type VerifierKey<G> = CommitterKey<G>;

impl<G: AffineCurve> PCVerifierKey for VerifierKey<G> {
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    fn supported_degree(&self) -> usize {
        self.comm_key.len() - 1
    }
}

/// Nothing to do to prepare this verifier key (for now).
pub type PreparedVerifierKey<G> = VerifierKey<G>;

impl<G: AffineCurve> PCPreparedVerifierKey<VerifierKey<G>> for PreparedVerifierKey<G> {
    /// prepare `PreparedVerifierKey` from `VerifierKey`
    fn prepare(vk: &VerifierKey<G>) -> Self {
        vk.clone()
    }
}

/// Commitment to a polynomial that optionally enforces a degree bound.
#[derive(Derivative)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct Commitment<G: AffineCurve> {
    /// A Pedersen commitment to the polynomial.
    pub comm: Vec<G>,

    /// A Pedersen commitment to the shifted polynomial.
    /// This is `none` if the committed polynomial does not
    /// enforce a strict degree bound.
    pub shifted_comm: Option<G>,
}

impl<G: AffineCurve> CanonicalSerialize for Commitment<G>
{
    fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError>
    {
        // More than enough for practical applications
        let len = u8::try_from(self.comm.len()).map_err(|_| SerializationError::NotEnoughSpace)?;
        CanonicalSerialize::serialize(&len, &mut writer)?;

        // Save only one of the coordinates of the point and one byte of flags in order
        // to be able to reconstruct the other coordinate
        for c in self.comm.iter() {
            CanonicalSerialize::serialize(c, &mut writer)?;
        }

        CanonicalSerialize::serialize(&self.shifted_comm, &mut writer)
    }

    fn serialized_size(&self) -> usize {
        1
            + self.comm.len() * self.comm[0].serialized_size()
            + self.shifted_comm.serialized_size()
    }
}

impl<G: AffineCurve> CanonicalDeserialize for Commitment<G> {
    fn deserialize<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        // Read comm
        let len: u8 = CanonicalDeserialize::deserialize(&mut reader)?;
        let mut comm = Vec::with_capacity(len as usize);
        for _ in 0..(len as usize) {
            let c: G = CanonicalDeserialize::deserialize(&mut reader)?;
            comm.push(c);
        }

        // Read shifted comm
        let shifted_comm: Option<G> = CanonicalDeserialize::deserialize(&mut reader)?;

        Ok(Self { comm, shifted_comm })
    }
}

impl<G: AffineCurve> PCCommitment for Commitment<G> {
    #[inline]
    fn empty() -> Self {
        Commitment {
            comm: vec![G::zero()],
            shifted_comm: None,
        }
    }

    fn has_degree_bound(&self) -> bool {
        false
    }
}

/// Nothing to do to prepare this commitment (for now).
pub type PreparedCommitment<E> = Commitment<E>;

impl<G: AffineCurve> PCPreparedCommitment<Commitment<G>> for PreparedCommitment<G> {
    /// prepare `PreparedCommitment` from `Commitment`
    fn prepare(vk: &Commitment<G>) -> Self {
        vk.clone()
    }
}

/// `Randomness` hides the polynomial inside a commitment and is outputted by `InnerProductArg::commit`.
#[derive(Derivative)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Randomness<G: AffineCurve> {
    /// Randomness is some scalar field element.
    pub rand: Vec<G::ScalarField>,

    /// Randomness applied to the shifted commitment is some scalar field element.
    pub shifted_rand: Option<G::ScalarField>,
}

impl<G: AffineCurve> PCRandomness for Randomness<G> {
    fn empty(segments_count: usize) -> Self {
        Self {
            rand: vec![G::ScalarField::zero(); segments_count],
            shifted_rand: None,
        }
    }

    fn rand<R: RngCore>(segments_count: usize, has_degree_bound: bool, rng: &mut R) -> Self {
        let rand = (0..segments_count).map(|_| G::ScalarField::rand(rng)).collect::<Vec<_>>();
        let shifted_rand = if has_degree_bound {
            Some(G::ScalarField::rand(rng))
        } else {
            None
        };

        Self { rand, shifted_rand }
    }
}

/// `Proof` is an evaluation proof that is output by `InnerProductArg::open`.
#[derive(Derivative)]
#[derivative(
Default(bound = ""),
Hash(bound = ""),
Clone(bound = ""),
Debug(bound = ""),
Eq(bound = ""),
PartialEq(bound = ""),
)]
pub struct Proof<G: AffineCurve> {
    /// Vector of left elements for each of the log_d iterations in `open`
    pub l_vec: Vec<G>,

    /// Vector of right elements for each of the log_d iterations within `open`
    pub r_vec: Vec<G>,

    /// Committer key from the last iteration within `open`
    pub final_comm_key: G,

    /// Coefficient from the last iteration within open`
    pub c: G::ScalarField,

    /// Commitment to the blinding polynomial.
    pub hiding_comm: Option<G>,

    /// Linear combination of all the randomness used for commitments
    /// to the opened polynomials, along with the randomness used for the
    /// commitment to the hiding polynomial.
    pub rand: Option<G::ScalarField>,
}

impl<G: AffineCurve> CanonicalSerialize for Proof<G> {
    fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {

        // l_vec
        // More than enough for practical applications
        let l_vec_len = u8::try_from(self.l_vec.len()).map_err(|_| SerializationError::NotEnoughSpace)?;
        CanonicalSerialize::serialize(&l_vec_len, &mut writer)?;

        // Save only one of the coordinates of the point and one byte of flags in order
        // to be able to reconstruct the other coordinate
        for p in self.l_vec.iter() {
            CanonicalSerialize::serialize(p, &mut writer)?;
        }

        // r_vec
        // More than enough for practical applications
        let r_vec_len = u8::try_from(self.r_vec.len()).map_err(|_| SerializationError::NotEnoughSpace)?;
        CanonicalSerialize::serialize(&r_vec_len, &mut writer)?;

        // Save only one of the coordinates of the point and one byte of flags in order
        // to be able to reconstruct the other coordinate
        for p in self.r_vec.iter() {
            CanonicalSerialize::serialize(p, &mut writer)?;
        }

        // Serialize the other fields
        CanonicalSerialize::serialize(&self.final_comm_key, &mut writer)?;
        CanonicalSerialize::serialize(&self.c, &mut writer)?;
        CanonicalSerialize::serialize(&self.hiding_comm, &mut writer)?;
        CanonicalSerialize::serialize(&self.rand, &mut writer)
    }

    fn serialized_size(&self) -> usize {
        1 + self.l_vec.iter().map(|item| item.serialized_size()).sum::<usize>()
            + 1 + self.r_vec.iter().map(|item| item.serialized_size()).sum::<usize>()
            + self.final_comm_key.serialized_size()
            + self.c.serialized_size()
            + self.hiding_comm.serialized_size()
            + self.rand.serialized_size()
    }
}

impl<G: AffineCurve> CanonicalDeserialize for Proof<G> {
    fn deserialize<R: Read>(mut reader: R) -> Result<Self, SerializationError> {

        // Read l_vec
        let l_vec_len: u8 = CanonicalDeserialize::deserialize(&mut reader)?;
        let mut l_vec = Vec::with_capacity(l_vec_len as usize);
        for _ in 0..(l_vec_len as usize) {
            let c: G = CanonicalDeserialize::deserialize(&mut reader)?;
            l_vec.push(c);
        }

        // Read r_vec
        let r_vec_len: u8 = CanonicalDeserialize::deserialize(&mut reader)?;
        let mut r_vec = Vec::with_capacity(r_vec_len as usize);
        for _ in 0..(r_vec_len as usize) {
            let c: G = CanonicalDeserialize::deserialize(&mut reader)?;
            r_vec.push(c);
        }

        // Read other fields
        let final_comm_key: G = CanonicalDeserialize::deserialize(&mut reader)?;
        let c: G::ScalarField = CanonicalDeserialize::deserialize(&mut reader)?;
        let hiding_comm: Option<G> = CanonicalDeserialize::deserialize(&mut reader)?;
        let rand: Option<G::ScalarField> = CanonicalDeserialize::deserialize(&mut reader)?;

        Ok(Self { l_vec, r_vec, final_comm_key, c, hiding_comm, rand })
    }
}

/// BatchProof generated by batching scheme
/// Boneh, et al. 2020, "Efficient polynomial commitment schemes for multiple points and polynomials",
/// IACR preprint 2020/81 https://eprint.iacr.org/2020/081
#[derive(Derivative)]
#[derivative(
Default(bound = ""),
Hash(bound = ""),
Clone(bound = ""),
Debug(bound = ""),
Eq(bound = ""),
PartialEq(bound = ""),
)]
pub struct BatchProof<G: AffineCurve> {

    /// This is a "classical" single-point multi-poly proof which involves all commitments:
    /// commitments from the initial claim and the new "batch_commitment"
    pub proof: Proof<G>,

    /// Commitment of the h(X) polynomial
    pub batch_commitment: Vec<G>,

    /// Values: v_i = p(x_i), where the query points x_i are not necessarily distinct.
    pub batch_values: BTreeMap<String, G::ScalarField>
}

impl<G: AffineCurve> CanonicalSerialize for BatchProof<G> {
    fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {

        // Serialize proof
        CanonicalSerialize::serialize(&self.proof, &mut writer)?;

        // Serialize batch_commitment
        // More than enough for practical applications
        let batch_commitment_len = u8::try_from(self.batch_commitment.len()).map_err(|_| SerializationError::NotEnoughSpace)?;
        CanonicalSerialize::serialize(&batch_commitment_len, &mut writer)?;

        // Save only one of the coordinates of the point and one byte of flags in order
        // to be able to reconstruct the other coordinate
        for comm in self.batch_commitment.iter() {
            CanonicalSerialize::serialize(comm, &mut writer)?;
        }

        // Serialize batch values
        // More than enough for practical applications
        let batch_values_len = u8::try_from(self.batch_values.len()).map_err(|_| SerializationError::NotEnoughSpace)?;
        CanonicalSerialize::serialize(&batch_values_len, &mut writer)?;
        for (k, v) in self.batch_values.iter() {
            CanonicalSerialize::serialize(k, &mut writer)?;
            CanonicalSerialize::serialize(v, &mut writer)?;
        }

        Ok(())
    }

    fn serialized_size(&self) -> usize {
        self.proof.serialized_size()
            + 1
            + self.batch_commitment.iter().map(|item| item.serialized_size()).sum::<usize>()
            + 1
            + self.batch_values.iter().map(|(k, v)| k.serialized_size() + v.serialized_size()).sum::<usize>()
    }
}

impl<G: AffineCurve> CanonicalDeserialize for BatchProof<G> {
    fn deserialize<R: Read>(mut reader: R) -> Result<Self, SerializationError> {

        // Read proof
        let proof: Proof<G> = CanonicalDeserialize::deserialize(&mut reader)?;

        // Read batch commitment
        let batch_commitment_len: u8 = CanonicalDeserialize::deserialize(&mut reader)?;
        let mut batch_commitment = Vec::with_capacity(batch_commitment_len as usize);
        for _ in 0..(batch_commitment_len as usize) {
            let comm: G = CanonicalDeserialize::deserialize(&mut reader)?;
            batch_commitment.push(comm);
        }

        // Read batch values
        let batch_values_len: u8 = CanonicalDeserialize::deserialize(&mut reader)?;
        let mut batch_values = BTreeMap::new();
        for _ in 0..(batch_values_len as usize) {
            let k: String = CanonicalDeserialize::deserialize(&mut reader)?;
            let v: G::ScalarField = CanonicalDeserialize::deserialize(&mut reader)?;
            batch_values.insert(k, v);
        }

        Ok(Self { proof, batch_commitment, batch_values })
    }
}

/// `SuccinctCheckPolynomial` is a succinctly-representated polynomial
/// generated from the `log_d` random oracle challenges generated in `open`.
/// It has the special property that can be evaluated in `O(log_d)` time.
#[derive(Clone)]
pub struct SuccinctCheckPolynomial<F: PrimeField>(pub Vec<F>);

impl<F: PrimeField> SuccinctCheckPolynomial<F> {

    /// Slighlty optimized way to compute it, taken from
    /// [o1-labs/marlin](https://github.com/o1-labs/marlin/blob/master/dlog/commitment/src/commitment.rs#L175)
    fn _compute_succinct_poly_coeffs(&self, mut init_coeffs: Vec<F>) -> Vec<F> {
        let challenges = &self.0;
        let log_d = challenges.len();
        let mut k: usize = 0;
        let mut pow: usize = 1;
        for i in 1..1 << log_d {
            k += if i == pow { 1 } else { 0 };
            pow <<= if i == pow { 1 } else { 0 };
            init_coeffs[i] = init_coeffs[i - (pow >> 1)] * challenges[log_d - 1 - (k - 1)];
        }
        init_coeffs
    }

    /// Computes the coefficients of the underlying degree `d` polynomial.
    pub fn compute_coeffs(&self) -> Vec<F> {
        self._compute_succinct_poly_coeffs(vec![F::one(); 1 << self.0.len()])
    }

    /// Computes the coefficients of the underlying degree `d` polynomial, scaled by
    /// a factor `scale`.
    pub fn compute_scaled_coeffs(&self, scale: F) -> Vec<F> {
        self._compute_succinct_poly_coeffs(vec![scale; 1 << self.0.len()])
    }

    /// Evaluate `self` at `point` in time `O(log_d)`.
    pub fn evaluate(&self, point: F) -> F {
        let challenges = &self.0;
        let log_d = challenges.len();

        let mut product = F::one();
        for (i, challenge) in challenges.iter().enumerate() {
            let i = i + 1;
            let elem_degree: u64 = (1 << (log_d - i)) as u64;
            let elem = point.pow([elem_degree]);
            product *= &(F::one() + &(elem * challenge));
        }

        product
    }
}