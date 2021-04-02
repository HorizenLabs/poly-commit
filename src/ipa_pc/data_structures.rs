use crate::*;
use crate::{PCCommitterKey, PCVerifierKey, Vec};
use algebra::{Field, ToBytes, to_bytes, FromBytes, UniformRand, AffineCurve, PrimeField};
use std::vec;
use rand_core::RngCore;

/// `UniversalParams` are the universal parameters for the inner product arg scheme.
#[derive(Derivative)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""))]
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

impl<G: AffineCurve> ToBytes for CommitterKey<G> {
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        (self.comm_key.len() as u32).write(&mut writer)?;
        for item in self.comm_key.iter() {
            item.write(&mut writer)?;
        }
        self.h.write(&mut writer)?;
        self.s.write(&mut writer)?;
        (self.max_degree as u8).write(&mut writer)
    }
}

impl<G: AffineCurve> FromBytes for CommitterKey<G> {
    #[inline]
    fn read<Read: std::io::Read>(mut reader: Read) -> std::io::Result<CommitterKey<G>> {
        let count = u32::read(&mut reader)?;
        let mut comm_key = vec![];
        for _ in 0..count {
            comm_key.push(G::read(&mut reader)?);
        }
        let h = G::read(&mut reader)?;
        let s = G::read(&mut reader)?;
        let max_degree = u8::read(&mut reader)? as usize;
        Ok(CommitterKey::<G>{
            comm_key,
            h,
            s,
            max_degree,
        })
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

    fn size_in_bytes(&self) -> usize {
        (to_bytes![G::zero()].unwrap().len() / 2) * self.comm.len()
    }
}

impl<G: AffineCurve> ToBytes for Commitment<G> {
    #[inline]
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        self.comm.write(&mut writer)?;
        self.shifted_comm.write(&mut writer)?;
        Ok(())
    }
}

impl<G: AffineCurve> FromBytes for Commitment<G> {
    #[inline]
    fn read<Read: std::io::Read>(mut reader: Read) -> std::io::Result<Commitment<G>> {
        let comm = G::read(&mut reader)?;
        let shifted_comm = Option::<G>::read(&mut reader)?;
        Ok(Commitment::<G>{
            comm,
            shifted_comm
        })
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

impl<G: AffineCurve> ToBytes for Randomness<G>
{
    #[inline]
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        self.rand.write(&mut writer)?;
        self.shifted_rand.write(&mut writer)?;
        Ok(())
    }
}

impl<G: AffineCurve> FromBytes for Randomness<G>
{
    #[inline]
    fn read<Read: std::io::Read>(mut reader: Read) -> std::io::Result<Randomness<G>> {
        let rand = G::ScalarField::read(&mut reader)?;
        let shifted_rand = Option::<G::ScalarField>::read(&mut reader)?;
        Ok(Randomness::<G>{
            rand,
            shifted_rand
        })
    }
}

/// `Proof` is an evaluation proof that is output by `InnerProductArg::open`.
#[derive(Derivative)]
#[derivative(
Default(bound = ""),
Hash(bound = ""),
Clone(bound = ""),
Debug(bound = "")
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

impl<G: AffineCurve> PCProof for Proof<G> {
    fn size_in_bytes(&self) -> usize {
        to_bytes![self].unwrap().len()
    }
}

impl<G: AffineCurve> ToBytes for Proof<G> {
    #[inline]
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        (self.l_vec.len() as u32).write(&mut writer)?;
        for item in self.l_vec.iter() {
            item.write(&mut writer)?;
        }
        (self.r_vec.len() as u32).write(&mut writer)?;
        for item in self.r_vec.iter() {
            item.write(&mut writer)?;
        }
        self.final_comm_key.write(&mut writer)?;
        self.c.write(&mut writer)?;
        self.hiding_comm.write(&mut writer)?;
        self.rand.write(&mut writer)
    }
}

impl<G: AffineCurve> FromBytes for Proof<G> {
    #[inline]
    fn read<Read: std::io::Read>(mut reader: Read) -> std::io::Result<Proof<G>> {
        let mut l_vec = vec![];
        let count = u32::read(&mut reader)? as usize;
        for _ in 0..count {
            l_vec.push(G::read(&mut reader)?);
        }
        let mut r_vec = vec![];
        let count = u32::read(&mut reader)? as usize;
        for _ in 0..count {
            r_vec.push(G::read(&mut reader)?);
        }
        let final_comm_key = G::read(&mut reader)?;
        let c = G::ScalarField::read(&mut reader)?;
        let hiding_comm = Option::<G>::read(&mut reader)?;
        let rand = Option::<G::ScalarField>::read(&mut reader)?;
        Ok(Proof::<G>{
            l_vec,
            r_vec,
            final_comm_key,
            c,
            hiding_comm,
            rand
        })
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
Debug(bound = "")
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

impl<G: AffineCurve> BatchPCProof for BatchProof<G> {
    fn size_in_bytes(&self) -> usize {
        to_bytes![self].unwrap().len()
    }
}

impl<G: AffineCurve> ToBytes for BatchProof<G> {
    #[inline]
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        self.proof.write(&mut writer)?;
        self.batch_commitment.write(&mut writer)?;
        self.batch_values.values().collect::<Vec<&G::ScalarField>>().write(&mut writer)
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