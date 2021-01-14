use algebra::{PrimeField, ToConstraintField, FpParameters};
use primitives::{PoseidonParameters, PoseidonSBox, PoseidonHash, AlgebraicSponge};
use crate::fiat_shamir::FiatShamirRng;

/// PoseidonHashGadget-backed implementation of FiatShamirRngGadget
pub mod constraints;

//TODO: This primitive might be slightly sped-up by working at byte level instead of bit level;
//      However, we need some extra care and new gadgets in the circuit (like a FromBytes gadget)
//      and I'm worried that the bytes corresponding to REPR_SHAVE_BITS (that will be all 0),
//      might affect security (maybe not).
//      Still, in terms of number of constraints, working on bit level is slightly cheaper.
impl<F, ConstraintF, P, SB> FiatShamirRng<F, ConstraintF> for PoseidonHash<ConstraintF, P, SB>
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
        use primitives::crh::poseidon::parameters::bn382::BN382FrPoseidonHash;

        let rng = &mut ChaCha20Rng::from_seed([
            1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2]);
        let non_native_inputs = vec![Fq::rand(rng); 5];
        let native_inputs = vec![Fr::rand(rng); 5];
        let byte_inputs = "TEST_FS_RNG_WITH_BN_382".as_bytes();
        test_absorb_squeeze_vals::<Fq, _, BN382FrPoseidonHash>(
            non_native_inputs,
            native_inputs,
            byte_inputs,

            (
                Fq::new(BigInteger384([12016618990178755936, 113170861864162129, 4879244684047526131, 7536486198472387696, 3005199297160601854, 578436848701623138])),
                Fr::new(BigInteger384([14108808286947537994, 438659023141181977, 12200179818727041293, 17425304848456118616, 1469018988105606060, 185938356707883218])),
                Fq::new(BigInteger384([3750802089028588180, 3878171735908549370, 15753965755587790825, 1437313415694079221, 8105394497193498031, 2091707477236339369])),
            ),
            (
                Fq::new(BigInteger384([9349821909120850162, 2583893546482351157, 11318221738885769420, 5586224527060837185, 18384939629932888033, 1927801529777188223])),
                Fr::new(BigInteger384([12421529125307652624, 12189242976828362088, 15552638547580932756, 2943385393865250656, 2422254917648279048, 910156443955058421])),
                Fq::new(BigInteger384([883559833201629700, 2297720271747027573, 10293539329733317053, 11843502315766683749, 9971289791846452217, 2129217580708431512])),
            ),
            (
                Fq::new(BigInteger384([1802634151726318859, 8932725432101101824, 17438325308273051863, 9195168340551825907, 8660026696007698207, 737597761618078455])),
                Fr::new(BigInteger384([2254176659747291193, 10909760097727274831, 1525194543008054803, 16228547171091262680, 2057160796565236781, 2328269622461191271])),
                Fq::new(BigInteger384([11397267253093576887, 353312307280226882, 5804830282428411167, 12502402992812411140, 3473442075068307199, 700133953511294972])),
            )
        );

        test_squeeze_consistency::<Fq, _, BN382FrPoseidonHash>();
    }
}