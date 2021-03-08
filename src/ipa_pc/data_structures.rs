use crate::*;
use crate::{PCCommitterKey, PCVerifierKey, Vec};
use algebra::{Field, ToBytes, to_bytes, UniformRand, AffineCurve};
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
        self.comm_key.write(&mut writer)?;
        self.h.write(&mut writer)?;
        self.s.write(&mut writer)?;
        (self.max_degree as u8).write(&mut writer)
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
Copy(bound = ""),
Debug(bound = ""),
PartialEq(bound = ""),
Eq(bound = "")
)]
pub struct Commitment<G: AffineCurve> {
    /// A Pedersen commitment to the polynomial.
    pub comm: G,

    /// A Pedersen commitment to the shifted polynomial.
    /// This is `none` if the committed polynomial does not
    /// enforce a strict degree bound.
    pub shifted_comm: Option<G>,
}

impl<G: AffineCurve> PCCommitment for Commitment<G> {
    #[inline]
    fn empty() -> Self {
        Commitment {
            comm: G::zero(),
            shifted_comm: None,
        }
    }

    fn has_degree_bound(&self) -> bool {
        false
    }

    fn size_in_bytes(&self) -> usize {
        to_bytes![G::zero()].unwrap().len() / 2
    }
}

impl<G: AffineCurve> ToBytes for Commitment<G> {
    #[inline]
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        self.comm.write(&mut writer)?;
        let shifted_exists = self.shifted_comm.is_some();
        shifted_exists.write(&mut writer)?;
        self.shifted_comm
            .as_ref()
            .unwrap_or(&G::zero())
            .write(&mut writer)
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
    pub rand: G::ScalarField,

    /// Randomness applied to the shifted commitment is some scalar field element.
    pub shifted_rand: Option<G::ScalarField>,
}

impl<G: AffineCurve> PCRandomness for Randomness<G> {
    fn empty() -> Self {
        Self {
            rand: G::ScalarField::zero(),
            shifted_rand: None,
        }
    }

    fn rand<R: RngCore>(_num_queries: usize, has_degree_bound: bool, rng: &mut R) -> Self {
        let rand = G::ScalarField::rand(rng);
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
        self.l_vec.write(&mut writer)?;
        self.r_vec.write(&mut writer)?;
        self.final_comm_key.write(&mut writer)?;
        self.c.write(&mut writer)?;
        self.hiding_comm
            .as_ref()
            .unwrap_or(&G::zero())
            .write(&mut writer)?;
        self.rand
            .as_ref()
            .unwrap_or(&G::ScalarField::zero())
            .write(&mut writer)
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
    pub batch_commitment: G,

    /// Values: v_i = p_i(x), where x is fresh random challenge
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

/// This implements the public aggregator for the IPA/DLOG commitment scheme.
#[derive(Clone)]
pub struct DLogAccumulator<G: AffineCurve> {
    /// Final committer key after the DLOG reduction.
    pub(crate) g_final:    Commitment<G>,

    /// Challenges of the DLOG reduction.
    pub(crate) xi_s:       SuccinctCheckPolynomial<G::ScalarField>
}

impl<G: AffineCurve> PCAccumulator for DLogAccumulator<G>
{
    type Commitment = Commitment<G>;
    type SuccinctDescription = SuccinctCheckPolynomial<G::ScalarField>;
}

impl<G: AffineCurve> ToBytes for DLogAccumulator<G> {
    #[inline]
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        self.g_final.write(&mut writer)?;
        self.xi_s.write(&mut writer)
    }
}

/// `SuccinctCheckPolynomial` is a succinctly-representated polynomial
/// generated from the `log_d` random oracle challenges generated in `open`.
/// It has the special property that can be evaluated in `O(log_d)` time.
#[derive(Clone)]
pub struct SuccinctCheckPolynomial<F: Field>(pub Vec<F>);

impl<F: Field> SuccinctCheckPolynomial<F> {

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

impl<F: Field> ToBytes for SuccinctCheckPolynomial<F> {
    #[inline]
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        self.0.write(&mut writer)
    }
}