use crate::*;
use crate::{PCCommitterKey, PCVerifierKey, Vec};
use algebra::{
    Field, UniformRand, AffineCurve, PrimeField, ToBytes,
};
use std::{
    io::{ Read, Write }, vec, convert::TryFrom,
};
use rand_core::RngCore;

/// `UniversalParams` are the universal parameters for the inner product arg scheme.
#[derive(Derivative)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""), Eq(bound = ""), PartialEq(bound = ""))]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniversalParams<G: AffineCurve> {
    /// The key used to commit to polynomials.
    pub comm_key: Vec<G>,

    /// Some group generator.
    pub h: G,

    /// Some group generator specifically used for hiding.
    pub s: G,

    /// H(comm_key, h, s, max_degree)
    pub hash: Vec<u8>
}

impl<G: AffineCurve> PCUniversalParams for UniversalParams<G> {
    fn max_degree(&self) -> usize {
        self.comm_key.len() - 1
    }
    fn get_hash(&self) -> &[u8] { self.hash.as_slice() }
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

    /// H(max_degree_comm_key, h, s, max_degree)
    pub hash: Vec<u8>,
}

impl<G: AffineCurve> SemanticallyValid for CommitterKey<G> {

    // Technically this function is redundant, since the keys are generated
    // through a deterministic procedure starting from a public string.
    fn is_valid(&self) -> bool {
        self.comm_key.is_valid() &&
            self.h.is_valid() &&
            self.s.is_valid() &&
            PCCommitterKey::supported_degree(self) <= self.max_degree
    }
}

impl<G: AffineCurve> PCCommitterKey for CommitterKey<G> {
    fn max_degree(&self) -> usize {
        self.max_degree
    }
    fn supported_degree(&self) -> usize {
        self.comm_key.len() - 1
    }

    fn get_hash(&self) -> &[u8] {
        self.hash.as_slice()
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

    fn get_hash(&self) -> &[u8] {
        self.hash.as_slice()
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

/// 'Segmentized' commitment to a polynomial that optionally enforces a degree bound.
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
    /// As we use segmentation, a commitment consists of several single-segment commitments.
    pub comm: Vec<G>,

    /// The commitment of the shifted last segment polynomial, as needed for degree proofs.
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
        self.shifted_comm.is_some()
    }
}

impl<G: AffineCurve> ToBytes for Commitment<G> {
    #[inline]
    fn write<W: Write>(&self, writer: W) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};

        self.serialize_uncompressed(writer)
            .map_err(|e| Error::new(ErrorKind::Other, format!{"{:?}", e}))
    }
}

impl<G: AffineCurve> SemanticallyValid for Commitment<G> {
    fn is_valid(&self) -> bool {
        self.comm.is_valid() &&
            if self.shifted_comm.is_some() { self.shifted_comm.as_ref().unwrap().is_valid() } else { true }
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

/// `Proof` is a single-point multi-poly opening proof output by `InnerProductArg::open`.
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

    /// Coefficient from the last iteration within `open`
    pub c: G::ScalarField,

    /// Commitment to the blinding polynomial.
    pub hiding_comm: Option<G>,

    /// Linear combination of all the randomness used for commitments
    /// to the opened polynomials, along with the randomness used for the
    /// commitment to the hiding polynomial.
    pub rand: Option<G::ScalarField>,
}

impl<G: AffineCurve> SemanticallyValid for Proof<G> {
    fn is_valid(&self) -> bool {
        self.l_vec.is_valid() &&
            self.r_vec.is_valid() &&
            self.l_vec.len() == self.r_vec.len() &&
            self.final_comm_key.is_valid() &&
            self.c.is_valid() &&
            {
                if self.hiding_comm.is_some() {
                    self.hiding_comm.as_ref().unwrap().is_valid() && self.rand.is_some()
                } else {
                    self.rand.is_none()
                }
            } &&
            // No need to re-check the hiding comm as the && operator is short-circuit
            {
                if self.rand.is_some() {
                    self.rand.as_ref().unwrap().is_valid()
                } else {
                    true
                }
            }
    }
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

        // We know r_vec must be equal in size to l_vec, so no need to serialize it too
        assert_eq!(self.l_vec.len(), self.r_vec.len());

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
            + self.r_vec.iter().map(|item| item.serialized_size()).sum::<usize>()
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
        let r_vec_len = l_vec_len;
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

/// Multi-point multi-poly opening proof according to [[BDFG2020]](https://eprint.iacr.org/2020/081).
/// Contains an extra (segemented) commitment `batch_commitment` which cannot be reproduced by 
/// linear combinations.     
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

impl<G: AffineCurve> SemanticallyValid for BatchProof<G> {
    fn is_valid(&self) -> bool {
        self.proof.is_valid() &&
            self.batch_commitment.is_valid() &&
            self.batch_values.iter().all(|(_, v)| v.is_valid())
    }
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
            + 1 + (self.batch_commitment.len() * self.batch_commitment[0].serialized_size())
            + 1 + self.batch_values.iter().map(|(k, v)| k.serialized_size() + v.serialized_size()).sum::<usize>()
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

/// The `SuccinctCheckPolynomial` is the dlog reduction polynomial
///     h(X) = Product_{i=0}^{d-1} (1 + xi_{d-i} * X^{2^i}),
/// where (xi_1,...xi_d) are the challenges of the dlog reduction steps.
/// This polynomial has the special property that it has a succinct description
/// and can be evaluated in `O(log(degree))` time, and the final committer key
/// G_final can be computed via MSM from the its coefficients.
#[derive(Clone, Debug, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct SuccinctCheckPolynomial<F: PrimeField>(pub Vec<F>);

impl<F: PrimeField> SuccinctCheckPolynomial<F> {

    /// Slightly optimized way to compute it, taken from
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