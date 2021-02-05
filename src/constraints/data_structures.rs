use crate::{
    LCTerm, LinearCombination,
    PolynomialCommitment,
    constraints::PolynomialCommitmentGadget,
};
use algebra::{PrimeField, AffineCurve, Field};
use r1cs_std::{
    fields::nonnative::nonnative_field_gadget::NonNativeFieldGadget,
    prelude::*,
};
use r1cs_core::{ConstraintSystem, SynthesisError};
use std::{
    collections::{HashMap, HashSet},
    borrow::Borrow, cmp::Eq, cmp::PartialEq, hash::Hash, marker::Sized
};

/// Define the minimal interface of prepared allocated structures.
pub trait PrepareGadget<Unprepared, ConstraintF: PrimeField>: Sized {
    /// Prepare from an unprepared element.
    fn prepare<CS: ConstraintSystem<ConstraintF>>(
        cs: CS,
        unprepared: &Unprepared
    ) -> Result<Self, SynthesisError>;
}

/// A coefficient of `LinearCombination`.
#[derive(Clone)]
pub enum LinearCombinationCoeffGadget<SimulationF: PrimeField, ConstraintF: PrimeField> {
    /// Coefficient 1.
    One,
    /// Coefficient -1.
    MinusOne,
    /// Other coefficient, represented as a nonnative field element.
    Gadget(NonNativeFieldGadget<SimulationF, ConstraintF>),
}

/// An allocated version of `LinearCombination`.
#[derive(Clone)]
pub struct LinearCombinationGadget<SimulationF: PrimeField, ConstraintF: PrimeField> {
    /// The label.
    pub label: String,
    /// The linear combination of `(coeff, poly_label)` pairs.
    pub terms: Vec<(LinearCombinationCoeffGadget<SimulationF, ConstraintF>, LCTerm)>,
}

/// A proof of satisfaction of linear combinations.
#[derive(Clone)]
pub struct BatchLCProofGadget<G: AffineCurve, PC: PolynomialCommitment<G>, PCG: PolynomialCommitmentGadget<G, PC>> {
    /// Evaluation proof.
    pub proof: PCG::BatchProofGadget,
    /// Evaluations required to verify the proof.
    pub evals: Option<Vec<NonNativeFieldGadget<G::ScalarField, <G::BaseField as Field>::BasePrimeField>>>,
}

impl<SimulationF: PrimeField, ConstraintF: PrimeField> AllocGadget<LinearCombination<SimulationF>, ConstraintF>
for LinearCombinationGadget<SimulationF, ConstraintF>
{
    fn alloc<F, T, CS: ConstraintSystem<ConstraintF>>(mut cs: CS, f: F) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<LinearCombination<SimulationF>>
    {
        let LinearCombination { label, terms } = f()?.borrow().clone();

        let new_terms: Vec<(LinearCombinationCoeffGadget<SimulationF, ConstraintF>, LCTerm)> = terms
            .iter()
            .enumerate()
            .map(|(i, term)| {
                let (f, lc_term) = term;

                let fg = NonNativeFieldGadget::alloc(
                    cs.ns(|| format!("alloc LC term {}", i)),
                    || Ok(f)
                ).unwrap();

                (LinearCombinationCoeffGadget::Gadget(fg), lc_term.clone())
            })
            .collect();

        Ok(Self {
            label,
            terms: new_terms,
        })
    }

    fn alloc_input<F, T, CS: ConstraintSystem<ConstraintF>>(mut cs: CS, f: F) -> Result<Self, SynthesisError>
        where
            F: FnOnce() -> Result<T, SynthesisError>,
            T: Borrow<LinearCombination<SimulationF>>
    {
        let LinearCombination { label, terms } = f()?.borrow().clone();

        let new_terms: Vec<(LinearCombinationCoeffGadget<SimulationF, ConstraintF>, LCTerm)> = terms
            .iter()
            .enumerate()
            .map(|(i, term)| {
                let (f, lc_term) = term;

                let fg = NonNativeFieldGadget::alloc_input(
                    cs.ns(|| format!("alloc LC term {}", i)),
                    || Ok(f)
                ).unwrap();

                (LinearCombinationCoeffGadget::Gadget(fg), lc_term.clone())
            })
            .collect();

        Ok(Self {
            label,
            terms: new_terms,
        })
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
/// A labeled point variable, for queries to a polynomial commitment.
pub struct LabeledPointGadget<SimulationF: PrimeField, ConstraintF: PrimeField>(
    /// The label of the point.
    /// MUST be a unique identifier in a query set.
    pub String,
    /// The point value.
    pub NonNativeFieldGadget<SimulationF, ConstraintF>
);

/// An allocated version of `QuerySet`.
#[derive(Clone)]
pub struct QuerySetGadget<SimulationF: PrimeField, ConstraintF: PrimeField>(
    pub HashSet<(String, LabeledPointGadget<SimulationF, ConstraintF>)>
);

/// An allocated version of `Evaluations`.
#[derive(Clone)]
pub struct EvaluationsGadget<SimulationF: PrimeField, ConstraintF: PrimeField>(
    pub HashMap<LabeledPointGadget<SimulationF, ConstraintF>, NonNativeFieldGadget<SimulationF, ConstraintF>>
);

impl<SimulationF: PrimeField, ConstraintF: PrimeField> EvaluationsGadget<SimulationF, ConstraintF> {
    /// find the evaluation result
    pub fn get_lc_eval(
        &self,
        lc_string: &str,
        point: &NonNativeFieldGadget<SimulationF, ConstraintF>,
    ) -> Result<NonNativeFieldGadget<SimulationF, ConstraintF>, SynthesisError> {
        let key = LabeledPointGadget::<SimulationF, ConstraintF>(
            String::from(lc_string), point.clone(),
        );
        Ok(self.0.get(&key).map(|v| (*v).clone()).unwrap())
    }
}