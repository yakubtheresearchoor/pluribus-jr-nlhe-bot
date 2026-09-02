fn main() {
    let bp = solver_core::blueprint::ShardedConnBlueprint::load("blueprint_conn_v5", 6, 5, 200, 16, 16, 7).unwrap();
    let c = |s: &str| solver_core::card::card_from_str(s).unwrap();
    let open: Vec<(u8,i32)> = vec![(0,0),(0,0),(0,0),(4,5)];
    let vs3: Vec<(u8,i32)> = vec![(0,0),(0,0),(0,0),(4,5),(4,15)];
    for (name, h, hist) in [
        ("J9s SB vs open", (c("Jd"), c("9d")), &open),
        ("AA  SB vs open", (c("Ad"), c("Ac")), &open),
        ("72o SB vs open", (c("7c"), c("2d")), &open),
        ("94s vs 3bet   ", (c("9h"), c("4h")), &vs3),
        ("AA  vs 3bet   ", (c("Ad"), c("Ac")), &vs3),
    ] {
        println!("{name}: row mass {:?}", bp.preflop_row_mass(h, hist));
    }
}
