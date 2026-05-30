// Metal Shading Language definitions for the flat tree structure.
// Must match the Rust repr(C) layout exactly.

#ifndef FLAT_TREE_METAL_H
#define FLAT_TREE_METAL_H

#include <metal_stdlib>
using namespace metal;

// Matches Rust FlatNode (20 bytes, 4-byte aligned, repr(C))
// struct FlatNode {
//     node_type: u8,      // offset 0
//     player_id: u8,      // offset 1
//     board_state: u8,    // offset 2
//     (implicit 1 byte padding)
//     num_children: u16,  // offset 4
//     children_start: u32, // offset 8
//     amount: i32,        // offset 12
//     action_label: u8,   // offset 16
//     (implicit 3 bytes padding for alignment)
// };

struct FlatNode {
    uint8_t node_type;       // 0
    uint8_t player_id;       // 1
    uint8_t board_state;     // 2
    uint8_t _pad0;           // 3 (padding)
    uint16_t num_children;   // 4-5
    uint16_t _pad1;          // 6-7 (padding)
    uint32_t children_start; // 8-11
    int32_t amount;          // 12-15
    uint8_t action_label;    // 16
    uint8_t _pad2[3];        // 17-19 (padding)
};

// Node type constants
constant uint8_t NODE_TYPE_TERMINAL = 0;
constant uint8_t NODE_TYPE_CHANCE   = 1;
constant uint8_t NODE_TYPE_PLAYER   = 2;

// Max actions per infoset
constant int MAX_NA = 4;

// Sentinel for "no infoset"
constant uint32_t UNUSED_U32 = 0xFFFFFFFFu;

// Action labels
constant uint8_t ACTION_LABEL_FOLD   = 0;
constant uint8_t ACTION_LABEL_CHECK  = 1;
constant uint8_t ACTION_LABEL_CALL   = 2;
constant uint8_t ACTION_LABEL_BET    = 3;
constant uint8_t ACTION_LABEL_RAISE  = 4;
constant uint8_t ACTION_LABEL_ALLIN  = 5;
constant uint8_t ACTION_LABEL_CHANCE = 6;

#endif // FLAT_TREE_METAL_H
