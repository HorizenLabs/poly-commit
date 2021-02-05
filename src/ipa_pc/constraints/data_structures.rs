use algebra::{AffineCurve, Field};
use r1cs_std::groups::GroupGadget;
use r1cs_std::alloc::AllocGadget;
use crate::ipa_pc::{VerifierKey, Commitment, Proof, BatchProof};
use r1cs_core::{ConstraintSystem, SynthesisError};
use r1cs_std::ToBytesGadget;
use r1cs_std::bits::uint8::UInt8;
use crate::constraints::PrepareGadget;
use r1cs_std::to_field_gadget_vec::ToConstraintFieldGadget;
use r1cs_std::fields::fp::FpGadget;
use crate::{PolynomialLabel, LabeledCommitment};
use std::{
    borrow::Borrow, marker::PhantomData
};
use r1cs_std::fields::nonnative::nonnative_field_gadget::NonNativeFieldGadget;

/// `CommitterKey` is used to commit to, and create evaluation proofs for, a given
/// polynomial.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = "")
)]
pub struct VerifierKeyGadget<
    G: AffineCurve,
    GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
        + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
>
{
    /// The key used to commit to polynomials.
    pub comm_key: Vec<GG>,

    /// A random group generator.
    pub h: GG,

    /// A random group generator that is to be used to make
    /// a commitment hiding.
    pub s: GG,

    /*/// The maximum degree supported by the parameters
    /// this key was derived from.
    pub max_degree: usize,*/

    #[doc(hidden)]
    _group: PhantomData<G>,
}

impl<G, GG> AllocGadget<VerifierKey<G>, <G::BaseField as Field>::BasePrimeField> for VerifierKeyGadget<G, GG>
    where
        G: AffineCurve,
        GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
            + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
{
    fn alloc<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(mut cs: CS, f: F) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<VerifierKey<G>>
    {
        let t = f()?;
        let vk = t.borrow();

        // Alloc comm keys
        let mut comm_key = Vec::new();
        for (i, comm) in vk.comm_key.iter().enumerate() {
            let gen = GG::alloc_checked(
                cs.ns(|| format!("alloc gen {}", i)),
                || Ok(comm.into_projective())
            )?;
            comm_key.push(gen);
        }

        // Alloc h
        let h = GG::alloc_checked(
            cs.ns(|| "alloc h"),
            || Ok(vk.h.into_projective())
        )?;

        // Alloc s
        let s = GG::alloc_checked(
            cs.ns(|| "alloc s"),
            || Ok(vk.s.into_projective())
        )?;

        Ok( Self{ comm_key, h, s, _group: PhantomData } )
    }

    fn alloc_input<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        mut cs: CS,
        f: F
    ) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<VerifierKey<G>>
    {
        let t = f()?;
        let vk = t.borrow();

        // Alloc comm keys
        let mut comm_key = Vec::new();
        for (i, comm) in vk.comm_key.iter().enumerate() {
            let gen = GG::alloc_input(
                cs.ns(|| format!("alloc input gen {}", i)),
                || Ok(comm.into_projective())
            )?;
            comm_key.push(gen);
        }

        // Alloc h
        let h = GG::alloc_input(
            cs.ns(|| "alloc input h"),
            || Ok(vk.h.into_projective())
        )?;

        // Alloc s
        let s = GG::alloc_input(
            cs.ns(|| "alloc input s"),
            || Ok(vk.s.into_projective())
        )?;

        Ok( Self{ comm_key, h, s, _group: PhantomData } )
    }
}

impl<G, GG> ToBytesGadget<<G::BaseField as Field>::BasePrimeField> for VerifierKeyGadget<G, GG>
    where
        G: AffineCurve,
        GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
            + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
{
    fn to_bytes<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        &self,
        mut cs: CS
    ) -> Result<Vec<UInt8>, SynthesisError>
    {
        let mut comm_keys_bytes = Vec::new();
        for (i, comm) in self.comm_key.iter().enumerate() {
            let mut comm_bytes = comm.to_bytes(cs.ns(|| format!("comm {} to bytes", i)))?;
            comm_keys_bytes.append(&mut comm_bytes);
        }

        let mut h_bytes = self.h.to_bytes(cs.ns(|| "h to bytes"))?;
        comm_keys_bytes.append(&mut h_bytes);

        let mut s_bytes = self.s.to_bytes(cs.ns(|| "s to bytes"))?;
        comm_keys_bytes.append(&mut s_bytes);

        Ok(comm_keys_bytes)
    }

    fn to_bytes_strict<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        &self,
        mut cs: CS
    ) -> Result<Vec<UInt8>, SynthesisError>
    {
        let mut comm_keys_bytes = Vec::new();
        for (i, comm) in self.comm_key.iter().enumerate() {
            let mut comm_bytes = comm.to_bytes_strict(cs.ns(|| format!("comm {} to bytes strict", i)))?;
            comm_keys_bytes.append(&mut comm_bytes);
        }

        let mut h_bytes = self.h.to_bytes_strict(cs.ns(|| "h to bytes strict"))?;
        comm_keys_bytes.append(&mut h_bytes);

        let mut s_bytes = self.s.to_bytes_strict(cs.ns(|| "s to bytes strict"))?;
        comm_keys_bytes.append(&mut s_bytes);

        Ok(comm_keys_bytes)
    }
}

impl<G, GG> ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField> for VerifierKeyGadget<G, GG>
    where
        G: AffineCurve,
        GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
            + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
{
    type FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>;

    fn to_field_gadget_elements<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        &self,
        mut cs: CS
    ) -> Result<Vec<Self::FieldGadget>, SynthesisError>
    {
        let mut fes = Vec::new();

        for (i, g) in self.comm_key.iter().enumerate() {
            let mut g_to_fes = g.to_field_gadget_elements(cs.ns(|| format!("gen {} to field elements", i)))?;
            fes.append(&mut g_to_fes);
        }

        let mut h_to_fes = self.h.to_field_gadget_elements(cs.ns(|| "h to field elements"))?;
        fes.append(&mut h_to_fes);

        let mut s_to_fes = self.s.to_field_gadget_elements(cs.ns(|| "s to field elements"))?;
        fes.append(&mut s_to_fes);

        Ok(fes)
    }
}

/// Nothing to do to prepare this verifier key (for now).
pub type PreparedVerifierKeyGadget<G, GG> = VerifierKeyGadget<G, GG>;

impl<
    G: AffineCurve,
    GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
        + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
>
PrepareGadget<VerifierKeyGadget<G, GG>, <G::BaseField as Field>::BasePrimeField>
for PreparedVerifierKeyGadget<G, GG>
{
    fn prepare<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        _cs: CS,
        unprepared: &VerifierKeyGadget<G, GG>
    ) -> Result<Self, SynthesisError>
    {
        Ok(unprepared.clone())
    }
}

/// Commitment to a polynomial that optionally enforces a degree bound.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct CommitmentGadget<
    G: AffineCurve,
    GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
        + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
> {
    /// A Pedersen commitment to the polynomial.
    pub comm: GG,

    /// A Pedersen commitment to the shifted polynomial.
    /// This is `none` if the committed polynomial does not
    /// enforce a strict degree bound.
    pub shifted_comm: Option<GG>,

    #[doc(hidden)]
    _group: PhantomData<G>,
}

impl<G, GG> AllocGadget<Commitment<G>, <G::BaseField as Field>::BasePrimeField> for CommitmentGadget<G, GG>
    where
        G: AffineCurve,
        GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
            + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
{
    fn alloc<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        mut cs: CS,
        f: F
    ) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<Commitment<G>>
    {
        f().and_then(|commitment| {
            let commitment = *commitment.borrow();
            let comm = commitment.comm;
            let comm_gadget = GG::alloc_checked(cs.ns(|| "alloc comm"), || Ok(comm.into_projective()))?;

            let shifted_comm = commitment.shifted_comm;
            let shifted_comm_gadget = if let Some(shifted_comm) = shifted_comm {
                Some(GG::alloc_checked(cs.ns(|| "alloc shifted comm"), || Ok(shifted_comm.into_projective()))?)
            } else {
                None
            };

            Ok(Self {
                comm: comm_gadget,
                shifted_comm: shifted_comm_gadget,
                _group: PhantomData,
            })
        })
    }

    fn alloc_input<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        mut cs: CS,
        f: F
    ) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<Commitment<G>>
    {
        f().and_then(|commitment| {
            let commitment = *commitment.borrow();
            let comm = commitment.comm;
            let comm_gadget = GG::alloc_input(cs.ns(|| "alloc input comm"), || Ok(comm.into_projective()))?;

            let shifted_comm = commitment.shifted_comm;
            let shifted_comm_gadget = if let Some(shifted_comm) = shifted_comm {
                Some(GG::alloc_input(cs.ns(|| "alloc input shifted comm"), || Ok(shifted_comm.into_projective()))?)
            } else {
                None
            };

            Ok(Self {
                comm: comm_gadget,
                shifted_comm: shifted_comm_gadget,
                _group: PhantomData,
            })
        })
    }
}

impl<G, GG> ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField> for CommitmentGadget<G, GG>
    where
        G: AffineCurve,
        GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
        + ToConstraintFieldGadget<
            <G::BaseField as Field>::BasePrimeField,
            FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>
        >
{
    type FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>;

    fn to_field_gadget_elements<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        &self,
        mut cs: CS
    ) -> Result<Vec<Self::FieldGadget>, SynthesisError>
    {
        let mut fes = Vec::new();

        let mut comm_to_fes = self.comm.to_field_gadget_elements(cs.ns(|| "comm to field elements"))?;
        fes.append(&mut comm_to_fes);

        if self.shifted_comm.is_some() {
            let mut shifted_comm_to_fes = self
                .shifted_comm
                .as_ref()
                .unwrap()
                .to_field_gadget_elements(cs.ns(|| "shifted comm to field elements"))?;
            fes.append(&mut shifted_comm_to_fes);
        }

        Ok(fes)
    }
}

impl<G, GG> ToBytesGadget<<G::BaseField as Field>::BasePrimeField> for CommitmentGadget<G, GG>
    where
        G: AffineCurve,
        GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
            + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
{
    fn to_bytes<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        &self,
        mut cs: CS
    ) -> Result<Vec<UInt8>, SynthesisError>
    {
        let mut comm_bytes = self.comm.to_bytes(
            cs.ns(|| "comm to bytes")
        )?;

        if self.shifted_comm.is_some() {
            let mut shifted_comm_bytes = self
                .shifted_comm
                .as_ref()
                .unwrap()
                .to_bytes(cs.ns(|| "shifted comm to bytes"))?;
            comm_bytes.append(&mut shifted_comm_bytes);
        }

        Ok(comm_bytes)
    }

    fn to_bytes_strict<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        &self,
        mut cs: CS
    ) -> Result<Vec<UInt8>, SynthesisError>
    {
        let mut comm_bytes = self.comm.to_bytes_strict(
            cs.ns(|| "comm to bytes strict")
        )?;

        if self.shifted_comm.is_some() {
            let mut shifted_comm_bytes = self
                .shifted_comm
                .as_ref()
                .unwrap()
                .to_bytes_strict(cs.ns(|| "shifted comm to bytes strict"))?;
            comm_bytes.append(&mut shifted_comm_bytes);
        }

        Ok(comm_bytes)
    }
}

/// Nothing to do to prepare this commitment (for now).
pub type PreparedCommitmentGadget<G, GG> = CommitmentGadget<G, GG>;

impl<
    G: AffineCurve,
    GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
    + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
>
PrepareGadget<CommitmentGadget<G, GG>, <G::BaseField as Field>::BasePrimeField>
for PreparedCommitmentGadget<G, GG>
{
    fn prepare<CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        _cs: CS,
        unprepared: &CommitmentGadget<G, GG>
    ) -> Result<Self, SynthesisError>
    {
        Ok(unprepared.clone())
    }
}

#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct LabeledCommitmentGadget<
    G: AffineCurve,
    GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
    + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
>
{
    pub(crate) label:        PolynomialLabel,
    pub(crate) commitment:   CommitmentGadget<G, GG>,
    pub(crate) degree_bound: Option<FpGadget<<G::BaseField as Field>::BasePrimeField>>,
}

impl<G, GG> AllocGadget<LabeledCommitment<Commitment<G>>, <G::BaseField as Field>::BasePrimeField> for LabeledCommitmentGadget<G, GG>
    where
        G: AffineCurve,
        GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
        + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>,
{
    fn alloc<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(mut cs: CS, f: F) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<LabeledCommitment<Commitment<G>>>
    {
        f().and_then(|labeled_comm| {
            let labeled_commitment = labeled_comm.borrow().clone();
            let label = labeled_commitment.label().to_string();
            let commitment = labeled_commitment.commitment().clone();
            let degree_bound = labeled_commitment.degree_bound().clone();

            let commitment = CommitmentGadget::<G, GG>::alloc_checked(
                cs.ns(|| "alloc commitment"),
                || Ok(commitment)
            )?;

            let degree_bound = if let Some(degree_bound) = degree_bound {
                Some(FpGadget::<<G::BaseField as Field>::BasePrimeField>::alloc(
                    cs.ns(|| "alloc degree bound"),
                    || Ok(<G::BaseField as Field>::BasePrimeField::from(degree_bound as u128))
                )?)
            } else {
                None
            };

            Ok(Self {
                label,
                commitment,
                degree_bound
            })
        })
    }

    fn alloc_input<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(mut cs: CS, f: F) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<LabeledCommitment<Commitment<G>>>
    {
        f().and_then(|labeled_comm| {
            let labeled_commitment = labeled_comm.borrow().clone();
            let label = labeled_commitment.label().to_string();
            let commitment = labeled_commitment.commitment();
            let degree_bound = labeled_commitment.degree_bound();

            let commitment = CommitmentGadget::<G, GG>::alloc_input(
                cs.ns(|| "alloc input commitment"),
                || Ok(commitment)
            )?;

            let degree_bound = if let Some(degree_bound) = degree_bound {
                Some(FpGadget::<<G::BaseField as Field>::BasePrimeField>::alloc_input(
                    cs.ns(|| "alloc input degree bound"),
                    || Ok(<G::BaseField as Field>::BasePrimeField::from(degree_bound as u128))
                )?)
            } else {
                None
            };

            Ok(Self {
                label,
                commitment,
                degree_bound
            })
        })
    }
}

/// Nothing to do to prepare this labeled commitment (for now).
pub type PreparedLabeledCommitmentGadget<G, GG> = LabeledCommitmentGadget<G, GG>;

/// `Proof` is an evaluation proof that is output by `InnerProductArg::open`.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = "")
)]
pub struct ProofGadget<
    G: AffineCurve,
    GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
    + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
>
{
    /// Vector of left elements for each of the log_d iterations in `open`
    pub l_vec: Vec<GG>,

    /// Vector of right elements for each of the log_d iterations within `open`
    pub r_vec: Vec<GG>,

    /// Committer key from the last iteration within `open`
    pub final_comm_key: GG,

    /// Coefficient from the last iteration within `open`
    pub c: NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>,

    /// Commitment to the blinding polynomial.
    pub hiding_comm: Option<GG>,

    /// Linear combination of all the randomness used for commitments
    /// to the opened polynomials, along with the randomness used for the
    /// commitment to the hiding polynomial.
    pub rand: Option<NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>>,
}

impl<G, GG> AllocGadget<Proof<G>, <G::BaseField as Field>::BasePrimeField> for ProofGadget<G, GG>
where
    G: AffineCurve,
    GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
    + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
{
    fn alloc<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        mut cs: CS,
        f: F
    ) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<Proof<G>>
    {
        f().and_then(|proof| {
            let Proof {
                l_vec,
                r_vec,
                final_comm_key,
                c,
                hiding_comm,
                rand
            } = proof.borrow().clone();

            let l_vec = l_vec.into_iter().enumerate().map(|(i, l)| {
                GG::alloc_checked(cs.ns(|| format!("alloc l {}", i)), || Ok(l.into_projective()))
            }).collect::<Result<Vec<_>, SynthesisError>>()?;

            let r_vec = r_vec.into_iter().enumerate().map(|(i, r)| {
                GG::alloc_checked(cs.ns(|| format!("alloc r {}", i)), || Ok(r.into_projective()))
            }).collect::<Result<Vec<_>, SynthesisError>>()?;

            let final_comm_key = GG::alloc_checked(
                cs.ns(|| "alloc final comm key"),
                || Ok(final_comm_key.into_projective())
            )?;

            let c = NonNativeFieldGadget::alloc(
                cs.ns(|| "alloc c"),
                || Ok(c)
            )?;

            let hiding_comm = if let Some(hiding_comm) = hiding_comm {
                Some(GG::alloc_checked(
                    cs.ns(|| "alloc hiding comm"),
                    || Ok(hiding_comm.into_projective()))?
                )
            } else {
                None
            };

            let rand = if let Some(rand) = rand {
                Some(NonNativeFieldGadget::alloc(cs.ns(|| "alloc rand"), || Ok(rand))?)
            } else {
                None
            };

            Ok(Self { l_vec, r_vec, final_comm_key, c, hiding_comm, rand })
        })
    }

    fn alloc_input<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(
        mut cs: CS,
        f: F
    ) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<Proof<G>>
    {
        f().and_then(|proof| {
            let Proof {
                l_vec,
                r_vec,
                final_comm_key,
                c,
                hiding_comm,
                rand
            } = proof.borrow().clone();

            let l_vec = l_vec.into_iter().enumerate().map(|(i, l)| {
                GG::alloc_input(cs.ns(|| format!("alloc input l {}", i)), || Ok(l.into_projective()))
            }).collect::<Result<Vec<_>, SynthesisError>>()?;

            let r_vec = r_vec.into_iter().enumerate().map(|(i, r)| {
                GG::alloc_input(cs.ns(|| format!("alloc input r {}", i)), || Ok(r.into_projective()))
            }).collect::<Result<Vec<_>, SynthesisError>>()?;

            let final_comm_key = GG::alloc_input(
                cs.ns(|| "alloc input final comm key"),
                || Ok(final_comm_key.into_projective())
            )?;

            let c = NonNativeFieldGadget::alloc_input(
                cs.ns(|| "alloc input c"),
                || Ok(c)
            )?;

            let hiding_comm = if let Some(hiding_comm) = hiding_comm {
                Some(GG::alloc_input(cs.ns(|| "alloc input hiding comm"), || Ok(hiding_comm.into_projective()))?)
            } else {
                None
            };

            let rand = if let Some(rand) = rand {
                Some(NonNativeFieldGadget::alloc_input(cs.ns(|| "alloc input rand"), || Ok(rand))?)
            } else {
                None
            };

            Ok(Self { l_vec, r_vec, final_comm_key, c, hiding_comm, rand })
        })
    }
}

/// Just a simple vector of ProofGadget (for now).
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
)]
pub struct BatchProofGadget<
    G: AffineCurve,
    GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
    + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
>(pub(crate) Vec<ProofGadget<G, GG>>);

impl<G, GG> AllocGadget<BatchProof<G>, <G::BaseField as Field>::BasePrimeField> for BatchProofGadget<G, GG>
    where
        G: AffineCurve,
        GG: GroupGadget<G::Projective, <G::BaseField as Field>::BasePrimeField>
        + ToConstraintFieldGadget<<G::BaseField as Field>::BasePrimeField, FieldGadget = FpGadget<<G::BaseField as Field>::BasePrimeField>>
{
    fn alloc<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(mut cs: CS, f: F) -> Result<Self, SynthesisError> where
        F: FnOnce() -> Result<T, SynthesisError>,
        T: Borrow<BatchProof<G>>
    {
        f().and_then(|proofs|{
            let proof_gadget_vec = proofs.borrow().clone().0.into_iter().enumerate().map(|(i, proof)| {
               ProofGadget::<G, GG>::alloc(cs.ns(|| format!("alloc proof {}", i)), || Ok(proof))
            }).collect::<Result<Vec<_>, SynthesisError>>()?;
            Ok(Self(proof_gadget_vec))
        })
    }

    fn alloc_input<F, T, CS: ConstraintSystem<<G::BaseField as Field>::BasePrimeField>>(mut cs: CS, f: F) -> Result<Self, SynthesisError> where
        F: FnOnce() -> Result<T, SynthesisError>,
        T: Borrow<BatchProof<G>>
    {
        f().and_then(|proofs|{
            let proof_gadget_vec = proofs.borrow().clone().0.into_iter().enumerate().map(|(i, proof)| {
                ProofGadget::<G, GG>::alloc_input(cs.ns(|| format!("alloc input proof {}", i)), || Ok(proof))
            }).collect::<Result<Vec<_>, SynthesisError>>()?;
            Ok(Self(proof_gadget_vec))
        })
    }
}