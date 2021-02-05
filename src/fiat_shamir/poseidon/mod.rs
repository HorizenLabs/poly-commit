use algebra::{PrimeField, ToConstraintField};
use primitives::{PoseidonParameters, PoseidonSBox, PoseidonSponge, AlgebraicSponge};
use crate::fiat_shamir::{FiatShamirRng, push_elements_to_sponge, get_elements_from_sponge};

/// PoseidonHashGadget-backed implementation of FiatShamirRngGadget
pub mod constraints;

impl<F, ConstraintF, P, SB> FiatShamirRng<F, ConstraintF> for PoseidonSponge<ConstraintF, P, SB>
where
    F: PrimeField,
    ConstraintF: PrimeField,
    P: PoseidonParameters<Fr = ConstraintF>,
    SB: PoseidonSBox<P>
{
    fn new() -> Self {
        <Self as AlgebraicSponge<ConstraintF>>::init()
    }

    fn absorb_nonnative_field_elements(&mut self, elems: &[F]) {
        push_elements_to_sponge::<F, _, _>(self, elems);
    }

    fn absorb_native_field_elements<T: ToConstraintField<ConstraintF>>(&mut self, elems: &[T]) {
        self.absorb(elems.iter().flat_map(|t| t.to_field_elements().unwrap()).collect())
    }

    fn absorb_bytes(&mut self, elems: &[u8]) {
        let mut bits = Vec::<bool>::new();
        for elem in elems.iter() {
            bits.append(&mut vec![
                elem & 128 != 0,
                elem & 64 != 0,
                elem & 32 != 0,
                elem & 16 != 0,
                elem & 8 != 0,
                elem & 4 != 0,
                elem & 2 != 0,
                elem & 1 != 0,
            ]);
        }
        let fes = bits.to_field_elements().unwrap();
        self.absorb(fes)
    }

    fn squeeze_nonnative_field_elements(&mut self, num: usize) -> Vec<F> {
        get_elements_from_sponge::<F, _, _>(self, num, false)
    }

    fn squeeze_native_field_elements(&mut self, num: usize) -> Vec<ConstraintF> {
        self.squeeze(num)
    }

    fn squeeze_128_bits_nonnative_field_elements(&mut self, num: usize) -> Vec<F> {
        get_elements_from_sponge::<F, _, _>(self, num, true)
    }
}

#[cfg(test)]
mod test {
    use algebra::{
        biginteger::BigInteger256,
        fields::tweedle::{
            fq::Fq, fr::Fr,
        }
    };
    use algebra::UniformRand;
    use crate::fiat_shamir::test::test_absorb_squeeze_vals;
    use crate::fiat_shamir::test::test_squeeze_consistency;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_fs_rng_poseidon_tweedle_fr(){

        use primitives::crh::poseidon::parameters::tweedle::TweedleFrPoseidonSponge;

        let rng = &mut ChaCha20Rng::from_seed([
            1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2]);
        let non_native_inputs = vec![Fq::rand(rng); 5];
        let native_inputs = vec![Fr::rand(rng); 5];
        let byte_inputs = "TEST_FS_RNG_WITH_TWEEDLE_FR".as_bytes();
        test_absorb_squeeze_vals::<Fq, _, TweedleFrPoseidonSponge>(
            non_native_inputs,
            native_inputs,
            byte_inputs,
            (
                Fq::new(BigInteger256([16516461252645333510, 10583677234309980696, 5655830494698526076, 2197463957596152184])),
                Fr::new(BigInteger256([1610033271994723242, 1420210711007191983, 416660917984504738, 791684900968741548])),
                Fq::new(BigInteger256([5894700794493033821, 9134426241895411758, 8257224035426028354, 3915305244924448883])),
            ),
            (
                Fq::new(BigInteger256([6456252896587977852, 12500533173042997806, 13956330679943382636, 4184773496251846477])),
                Fr::new(BigInteger256([8855038643308334620, 4634386296115890190, 4579257527088864698, 691747633033767970])),
                Fq::new(BigInteger256([6018885514446331441, 12572350802444592318, 468387694499328978, 3900333247289797302])),
            ),
            (
                Fq::new(BigInteger256([8102280768793942757, 7525476414605893128, 13706533693355979953, 1327905306094223431])),
                Fr::new(BigInteger256([9407600689200936381, 6574634219841608579, 2996693058326046885, 3521360952488081455])),
                Fq::new(BigInteger256([14731716235130337525, 4919963741794774586, 16578268229673767195, 3638084636820484437])),
            )
        );

        test_squeeze_consistency::<Fq, _, TweedleFrPoseidonSponge>();
    }

    #[test]
    fn test_fs_rng_poseidon_tweedle_fq(){

        use primitives::crh::poseidon::parameters::tweedle::TweedleFqPoseidonSponge;

        let rng = &mut ChaCha20Rng::from_seed([
            1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2]);
        let non_native_inputs = vec![Fr::rand(rng); 5];
        let native_inputs = vec![Fq::rand(rng); 5];
        let byte_inputs = "TEST_FS_RNG_WITH_TWEEDLE_FQ".as_bytes();
        test_absorb_squeeze_vals::<Fr, _, TweedleFqPoseidonSponge>(
            non_native_inputs,
            native_inputs,
            byte_inputs,
            (
                Fr::new(BigInteger256([10762946725029652877, 13005361501190432681, 18024519008534991755, 3248501990851186298])),
                Fq::new(BigInteger256([8941957288665901150, 18066340813446135570, 15284631204673323353, 2402757555543135154])),
                Fr::new(BigInteger256([6378365208770626185, 13420982113762396523, 15904242776272886228, 4262364034036701501])),
            ),
            (
                Fr::new(BigInteger256([261060505832591120, 6645565064699753870, 5681232411053781566, 2415417108750069081])),
                Fq::new(BigInteger256([15918289345942480957, 10678384469336226475, 9949997421033032654, 1781468515090487909])),
                Fr::new(BigInteger256([5336496322269491397, 9099152767105313647, 12181864234399154007, 3639884640497234796])),
            ),
            (
                Fr::new(BigInteger256([1616875255399882483, 18446592942995947804, 887637818947102451, 3885902919121541649])),
                Fq::new(BigInteger256([14893396250722415518, 14008025679778793897, 8676609536434381080, 731394336888061081])),
                Fr::new(BigInteger256([9205189026741700221, 12198393360744278704, 3357649801577616719, 4133680788571664179])),
            ),
        );

        test_squeeze_consistency::<Fr, _, TweedleFqPoseidonSponge>();
    }
}