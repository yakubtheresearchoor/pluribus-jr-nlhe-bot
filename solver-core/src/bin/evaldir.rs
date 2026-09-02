fn main() {
    use solver_core::hand::eval::Hand;
    // board 3c 8d Jh Ks 2s ; AA vs 72o (7h 2... use 7h 4d high-card)
    let b = |h: Hand, c: usize| h.add_card(c);
    let mut base = Hand::new();
    for &c in &[4usize, 24, 39, 47, 3] { base = b(base, c); } // 3,8,J,K,2 mixed suits
    let aa = b(b(base, 48), 49).evaluate_internal();  // AcAd -> pair of aces
    let hc = b(b(base, 20), 9).evaluate_internal();   // 7x 4x -> high card
    println!("AA-pair internal: {aa}");
    println!("high-card internal: {hc}");
    println!("bigger-is-better would need AA > high-card: {}", aa > hc);
}
