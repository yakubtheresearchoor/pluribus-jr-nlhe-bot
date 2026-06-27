//! Inspect a dumped EQR preflop chart (`<base>.f32` + `.json`): confirm the
//! action menu (14 sized raises + all-in) is in the tree, then print a full
//! 13×13 open-range grid for every RFI position (UTG→SB) so the chart can be
//! eyeballed against standard GTO. Read-only.
//!
//! Run: cargo run --release -p play-harness --bin eqr_dump -- preflop_eqr_bbfix

use play_harness::preflop_player::PreflopPlayer;
use solver_core::tree::flat::{
    ACTION_LABEL_ALLIN, ACTION_LABEL_CALL, ACTION_LABEL_FOLD, ACTION_LABEL_RAISE, MAX_NA_PREFLOP,
    NODE_TYPE_PLAYER,
};

// Card ranks high→low for the matrix (A=12 … 2=0); card = rank*4 + suit.
const RDISP: [u8; 13] = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
const RCH: [char; 13] = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

fn class_of(hi: u8, lo: u8, suited: bool, pair: bool) -> usize {
    let (c1, c2) = if pair {
        (hi * 4, hi * 4 + 1)
    } else if suited {
        (hi * 4, lo * 4)
    } else {
        (hi * 4, lo * 4 + 1)
    };
    PreflopPlayer::hand_class(c1, c2)
}

/// Open frequency (raise + all-in %) at `node` for a hand class.
fn open_pct(p: &PreflopPlayer, node: usize, labels: &[u8], cl: usize) -> f32 {
    let mut buf = [0f32; MAX_NA_PREFLOP];
    let n = p.action_dist(node, cl, &mut buf);
    (0..n)
        .filter(|&a| labels[a] == ACTION_LABEL_RAISE || labels[a] == ACTION_LABEL_ALLIN)
        .map(|a| buf[a])
        .sum::<f32>()
        * 100.0
}

fn print_grid(p: &PreflopPlayer, node: usize, label: &str) {
    let labels: Vec<u8> =
        p.tree.node_children(node).iter().map(|&c| p.tree.nodes[c as usize].action_label).collect();
    let na = labels.len();
    let nr = labels.iter().filter(|&&l| l == ACTION_LABEL_RAISE).count();
    let allin = labels.iter().any(|&l| l == ACTION_LABEL_ALLIN);
    // overall % of the 1326 combos opened (combo-weighted: pair=6, suited=4, offsuit=12)
    let mut opened = 0.0f64;
    for i in 0..13 {
        for j in 0..13 {
            let pair = i == j;
            let suited = j > i;
            let (hi, lo) = if suited { (RDISP[i], RDISP[j]) } else { (RDISP[j], RDISP[i]) };
            let cl = class_of(hi, lo, suited, pair);
            let w = if pair { 6.0 } else if suited { 4.0 } else { 12.0 };
            opened += w * open_pct(p, node, &labels, cl) as f64 / 100.0;
        }
    }
    println!(
        "\n=== {label}  (node {node}, na={na}, {nr} raise sizes{}) — opens {:.1}% of hands ===",
        if allin { "+allin" } else { "" },
        opened / 1326.0 * 100.0
    );
    print!("    ");
    for j in 0..13 {
        print!(" {} ", RCH[j]);
    }
    println!("   (upper-right=suited, lower-left=offsuit, diag=pairs)");
    for i in 0..13 {
        print!(" {}  ", RCH[i]);
        for j in 0..13 {
            let pair = i == j;
            let suited = j > i;
            let (hi, lo) = if suited { (RDISP[i], RDISP[j]) } else { (RDISP[j], RDISP[i]) };
            let cl = class_of(hi, lo, suited, pair);
            let o = open_pct(p, node, &labels, cl);
            if o < 0.5 {
                print!("  ·");
            } else if o > 99.0 {
                print!(" ██");
            } else {
                print!("{:3}", o.round() as i32);
            }
        }
        println!();
    }
}

fn main() {
    let base = std::env::args().nth(1).unwrap_or_else(|| "preflop_eqr_bbfix".into());
    let p = PreflopPlayer::load(&base).expect("load EQR chart");

    // MENU CENSUS.
    let (mut folds, mut calls, mut raises, mut allins, mut max_na) = (0u64, 0u64, 0u64, 0u64, 0usize);
    for n in 0..p.tree.num_nodes() {
        if p.tree.nodes[n].node_type != NODE_TYPE_PLAYER {
            continue;
        }
        max_na = max_na.max(p.tree.nodes[n].num_children as usize);
        for &c in p.tree.node_children(n) {
            match p.tree.nodes[c as usize].action_label {
                ACTION_LABEL_FOLD => folds += 1,
                ACTION_LABEL_CALL => calls += 1,
                ACTION_LABEL_RAISE => raises += 1,
                ACTION_LABEL_ALLIN => allins += 1,
                _ => {}
            }
        }
    }
    println!("MENU CENSUS ({base}): max_na={max_na} | fold={folds} call={calls} raise={raises} ALLIN={allins} edges");

    // RFI chain: from the opener, follow the FOLD edge to each next position's open.
    let pos = ["UTG", "HJ", "CO", "BTN", "SB"];
    let mut node = (0..p.tree.num_nodes())
        .find(|&n| p.tree.nodes[n].node_type == NODE_TYPE_PLAYER && p.tree.nodes[n].num_children > 1)
        .expect("opener");
    for (k, name) in pos.iter().enumerate() {
        print_grid(&p, node, &format!("{name} RFI"));
        // follow fold edge to the next opener
        let kids = p.tree.node_children(node);
        let fold_child = kids
            .iter()
            .copied()
            .find(|&c| p.tree.nodes[c as usize].action_label == ACTION_LABEL_FOLD);
        match fold_child {
            Some(fc) => {
                // descend through any chance/terminal until the next player decision
                let mut cur = fc as usize;
                let mut hops = 0;
                while hops < 20
                    && (p.tree.nodes[cur].node_type != NODE_TYPE_PLAYER
                        || p.tree.nodes[cur].num_children <= 1)
                {
                    if p.tree.node_children(cur).is_empty() {
                        break;
                    }
                    cur = p.tree.node_children(cur)[0] as usize;
                    hops += 1;
                }
                if p.tree.nodes[cur].node_type == NODE_TYPE_PLAYER && p.tree.nodes[cur].num_children > 1 {
                    node = cur;
                } else if k + 1 < pos.len() {
                    println!("\n(could not locate {} RFI node — stopping)", pos[k + 1]);
                    break;
                }
            }
            None => break,
        }
    }
}
