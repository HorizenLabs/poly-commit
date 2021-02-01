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
    use algebra::UniformRand;
    use crate::fiat_shamir::test::test_absorb_squeeze_vals;
    use crate::fiat_shamir::test::test_squeeze_consistency;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_fs_rng_poseidon_bn_382(){

        use algebra::{
            biginteger::BigInteger384,
            fields::bn_382::{
                fq::Fq, fr::Fr,
            }
        };
        use primitives::crh::poseidon::parameters::bn382::BN382FrPoseidonSponge;

        let rng = &mut ChaCha20Rng::from_seed([
            1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2]);
        let non_native_inputs = vec![Fq::rand(rng); 5];
        let native_inputs = vec![Fr::rand(rng); 5];
        let byte_inputs = "TEST_FS_RNG_WITH_BN_382".as_bytes();
        test_absorb_squeeze_vals::<Fq, _, BN382FrPoseidonSponge>(
            non_native_inputs,
            native_inputs,
            byte_inputs,

            (
                Fq::new(BigInteger384([16968539842303851061, 17801941139790967157, 11912872069032247438, 17347049997685724293, 13166518364374500900, 1501581078475130236])),
                Fr::new(BigInteger384([11009198143356571695, 9233983554336419381, 5206816659168252705, 4904572273100637839, 294675676069883180, 1073871216927475505])),
                Fq::new(BigInteger384([15482123150356760728, 10708171711392742622, 1687588660499287236, 16801333765857531051, 6655449532438561903, 1588057202412049339])),
            ),
            (
                Fq::new(BigInteger384([252899744532148700, 5167787041418753963, 4135228996322313624, 17778754025806233207, 8342627014844318588, 1539545592305393292])),
                Fr::new(BigInteger384([7197609493738603709, 16793498479102870848, 12116352465778914752, 7555252794275463479, 15762501231817493649, 1559517585302804411])),
                Fq::new(BigInteger384([725413744086744838, 14140917903316955258, 8097716872723691675, 12928722430078030776, 12458917807539945424, 326619231893636028])),
            ),
            (
                Fq::new(BigInteger384([3605268303452637718, 17865450864202203648, 16429906542836552110, 18390336681103651815, 17320053392015396414, 1475195523236156910])),
                Fr::new(BigInteger384([17327961317228976599, 14920233244743080316, 11337553424800437501, 10467062516926635978, 8332404348717572274, 967231975056851993])),
                Fq::new(BigInteger384([13263303052593867052, 632240603357477765, 3296826592338440535, 1361321415044672543, 15199457239458622347, 878356618377807551])),
            )
        );

        test_squeeze_consistency::<Fq, _, BN382FrPoseidonSponge>();
    }
}