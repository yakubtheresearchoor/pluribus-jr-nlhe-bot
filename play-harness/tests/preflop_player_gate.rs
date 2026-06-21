//! Validation gate for the EQR preflop loader (production-baseline piece 1):
//! the loaded strategy must produce a SANE UTG opening range — AA/KK raise,
//! 72o folds — confirming the tree rebuild + array layout are correct.
//!
//! Run: cargo test --release -p play-harness --test preflop_player_gate -- --ignored --nocapture

use play_harness::preflop_player::PreflopPlayer;
use solver_core::card::card_from_str;
use solver_core::tree::flat::MAX_NA_PREFLOP;

#[test]
#[ignore = "needs preflop_eqr_bbfix artifact (2GB); --ignored --nocapture --release"]
fn preflop_loader_utg_range_sane() {
    let base = std::env::var("PF_STRAT").unwrap_or_else(|_| "preflop_eqr_bbfix".into());
    if !std::path::Path::new(&format!("{base}.f32")).exists() {
        eprintln!("SKIP: no preflop artifact at {base}.f32");
        return;
    }
    let pp = PreflopPlayer::load(&base).unwrap();
    eprintln!("loaded preflop strategy: {} tree nodes", pp.tree.num_nodes());

    // UTG open = the FIRST player decision node (fold-chain starts at UTG).
    let utg = (0..pp.tree.num_nodes())
        .find(|&i| pp.tree.nodes[i].is_player())
        .expect("a preflop decision node");
    let labels: Vec<u8> = pp
        .tree
        .node_children(utg)
        .iter()
        .map(|&c| pp.tree.nodes[c as usize].action_label)
        .collect();
    eprintln!("UTG node {utg}: {} actions, labels {labels:?} (0=fold 4=raise)", labels.len());

    let fold_a = labels.iter().position(|&l| l == 0); // fold action index, if present
    let mut buf = [0f32; MAX_NA_PREFLOP];
    let mut probe = |name: &str, c1: &str, c2: &str| -> f32 {
        let h = PreflopPlayer::hand_class(card_from_str(c1).unwrap(), card_from_str(c2).unwrap());
        let na = pp.action_dist(utg, h, &mut buf);
        let fold_p = fold_a.map(|a| buf[a]).unwrap_or(0.0);
        let raise_p: f32 = (0..na).filter(|&a| labels[a] >= 3).map(|a| buf[a]).sum();
        eprintln!("  {name:<5} class={h:>3}: fold {fold_p:.3}  raise {raise_p:.3}");
        fold_p
    };

    let aa = probe("AA", "As", "Ah");
    let kk = probe("KK", "Ks", "Kh");
    probe("ATs", "As", "Ts");
    probe("98s", "9s", "8s");
    probe("K9o", "Ks", "9h");
    let t72o = probe("72o", "7s", "2c");
    let t32o = probe("32o", "3s", "2c");

    assert!(aa < 0.10, "AA should rarely fold UTG, got {aa}");
    assert!(kk < 0.10, "KK should rarely fold UTG, got {kk}");
    assert!(t72o > 0.85, "72o should mostly fold UTG, got {t72o}");
    assert!(t32o > 0.85, "32o should mostly fold UTG, got {t32o}");
    eprintln!("✓ UTG range sane — preflop loader validated");
}
