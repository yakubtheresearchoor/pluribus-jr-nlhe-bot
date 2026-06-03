/// Direct zero-sum check on side_pot_showdown_cfv
use solver_core::solver::showdown::side_pot_showdown_cfv;

#[test]
fn direct_zero_sum_3hands() {
    // 3 hands, 3 players, all non-conflicting cards, uniform reach
    let hand_cards: Vec<u8> = vec![0, 1,  2, 3,  4, 5]; // 3 hands with unique cards
    let nh = 3;
    let np = 3u8;

    // Sorted by strength: hand 0 > hand 1 > hand 2
    let sorted_opp_str: Vec<u16> = vec![1, 2, 3,  1, 2, 3]; // 2 opponents, same order
    let sorted_opp_idx: Vec<u16> = vec![2, 1, 0,  2, 1, 0]; // weakest first
    let sorted_pl_str: Vec<u16> = vec![1, 2, 3];
    let sorted_pl_idx: Vec<u16> = vec![2, 1, 0];

    let reach = vec![1.0f32; nh]; // uniform reach for each opponent
    let contributions = vec![5i32; np as usize];

    let mut cfv_sum = vec![0.0f32; nh];

    for traverser in 0..np as usize {
        let opp_reach: Vec<&[f32]> = (0..np as usize)
            .filter(|&p| p != traverser)
            .map(|_| reach.as_slice())
            .collect();

        let cfv = side_pot_showdown_cfv(
            &opp_reach, &hand_cards, nh,
            &sorted_opp_str, &sorted_opp_idx,
            &sorted_pl_str, &sorted_pl_idx,
            &contributions, 0, traverser, np, 15,
        );
        for h in 0..nh { cfv_sum[h] += cfv[h]; }
        eprintln!("Traverser {}: {:?}", traverser, cfv);
    }

    let total: f32 = cfv_sum.iter().sum();
    eprintln!("Per-hand sum: {:?}", cfv_sum);
    eprintln!("Total sum: {:.8}", total);
    eprintln!("Per-hand avg: {:.8}", total / nh as f32);

    assert!(total.abs() < 0.001, "Zero-sum violation: {}", total);
}
