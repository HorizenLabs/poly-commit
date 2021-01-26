use algebra::{PrimeField, ToConstraintField, FpParameters};
use primitives::{PoseidonParameters, PoseidonSBox, PoseidonSponge, AlgebraicSponge};
use crate::fiat_shamir::FiatShamirRng;

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
        <Self as AlgebraicSponge<ConstraintF>>::new()
    }

    fn absorb_nonnative_field_elements(&mut self, elems: &[F]) {
        // Serialize elems to bits
        let elems_bits: Vec<bool> = elems
            .iter()
            .flat_map(|fe|{ fe.write_bits() })
            .collect();

        // Pack (safely) elems_bits into native field elements
        let native_fes = elems_bits.as_slice().to_field_elements().unwrap();

        // Absorb native field elements
        self.absorb(native_fes);
    }

    fn absorb_native_field_elements<T: ToConstraintField<ConstraintF>>(&mut self, elems: &[T]) {
        self.absorb(elems.iter().flat_map(|t| t.to_field_elements().unwrap()).collect())
    }

    fn absorb_bytes(&mut self, elems: &[u8]) {
        let bits: Vec<bool> = elems
            .iter()
            .flat_map(|byte| primitives::bytes_to_bits(&[*byte]).iter().rev().cloned().collect::<Vec<bool>>())
            .collect();
        let fes = bits.to_field_elements().unwrap();
        self.absorb(fes)
    }

    fn squeeze_nonnative_field_elements(&mut self, num: usize) -> Vec<F> {
        // Compute number of native field elements we need in order to satisfy the request of
        // num non-native field elements as output
        let nonnative_bits = F::Params::CAPACITY as usize;
        let required_bits = num * nonnative_bits;
        let required_native_fes = {
            let num = (required_bits/ConstraintF::Params::CAPACITY as usize) as f64;
            if num == 0.0 { 1 } else { num.ceil() as usize }
        };

        // Squeeze required number of native field elements
        let native_squeeze = self.squeeze(required_native_fes);

        // Serialize them to bits
        let native_bits = native_squeeze
            .into_iter()
            // Take only capacity bits: this will avoid a range check in the corresponding gadget
            // This doesn't affect security as we are truncating the squeeze outputs.
            .flat_map(|native_fe: ConstraintF| { native_fe.write_bits() })
            .collect::<Vec<_>>();

        // Read non-native field elements from the bit vector, and return the required num
        native_bits.to_field_elements().unwrap().into_iter().take(num).collect()
    }

    fn squeeze_native_field_elements(&mut self, num: usize) -> Vec<ConstraintF> {
        self.squeeze(num)
    }

    fn squeeze_128_bits_nonnative_field_elements(&mut self, num: usize) -> Vec<F> {

        // Compute number of challenges we can extract from a single field element
        let modulus_bits = F::Params::MODULUS_BITS as usize;
        assert!(modulus_bits >= 128);
        let challenges_per_fe = modulus_bits/128;

        // Compute number of field elements we need in order to provide the requested 'num'
        // 128 bits field elements
        let to_squeeze = {
            let num = (num/challenges_per_fe) as f64;
            if num == 0.0 { 1 } else { num.ceil() as usize }
        };

        // Squeeze the required number of field elements
        let outputs_bits = self
            .squeeze_nonnative_field_elements(to_squeeze)
            .into_iter()
            .flat_map(|fe: F|{ fe.write_bits() }).collect::<Vec<_>>();

        // Take the required amount of 128 bits chunks and read (safely) a non-native field
        // element out of each of them.
        outputs_bits.chunks(128).take(num).map(|bits| {
            F::read_bits(bits.to_vec()).expect("Should be able to read a nonnativefield element from 128 bits")
        }).collect()
    }
}

#[cfg(test)]
mod test {
    use algebra::UniformRand;
    use crate::fiat_shamir::test::{
        test_absorb_squeeze_vals, test_squeeze_consistency
    };
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
                Fq::new(BigInteger384([2895660242738789017, 8512201728826116984, 10846410857391028522, 1177847072281288486, 1176675876650363556, 117059997667922578])),
                Fr::new(BigInteger384([7139029093582782450, 5122260996318787352, 11817825065862403011, 11032640916939218134, 3507534247076097382, 888689454973592979])),
                Fq::new(BigInteger384([18325121246728684406, 10073074744960290716, 5181761558013828819, 16377396632814968485, 12827244535173952658, 401369050975199809])),
            ),
            (
                Fq::new(BigInteger384([9349821909120850162, 2583893546482351157, 11318221738885769420, 5586224527060837185, 18384939629932888033, 1927801529777188223])),
                Fr::new(BigInteger384([7197609493738603709, 16793498479102870848, 12116352465778914752, 7555252794275463479, 15762501231817493649, 1559517585302804411])),
                Fq::new(BigInteger384([9404725472876462018, 17370287534333024298, 15862891673946816230, 9736815574430304622, 7197275726412521400, 1379329904283317430])),
            ),
            (
                Fq::new(BigInteger384([1802634151726318859, 8932725432101101824, 17438325308273051863, 9195168340551825907, 8660026696007698207, 737597761618078455])),
                Fr::new(BigInteger384([17327961317228976599, 14920233244743080316, 11337553424800437501, 10467062516926635978, 8332404348717572274, 967231975056851993])),
                Fq::new(BigInteger384([7927511781575854671, 13993118231894507329, 14686499907236610645, 10872235923201052429, 13401804413947996421, 1377617888218943465])),
            )
        );

        test_squeeze_consistency::<Fq, _, BN382FrPoseidonSponge>();
    }
}