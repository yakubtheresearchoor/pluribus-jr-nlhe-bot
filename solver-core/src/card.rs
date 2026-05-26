use std::mem;

pub type Card = u8;

pub const NOT_DEALT: Card = Card::MAX;
pub const NUM_CARDS_IN_DECK: usize = 52;

#[inline]
pub fn card_pair_to_index(mut card1: Card, mut card2: Card) -> usize {
    if card1 > card2 {
        mem::swap(&mut card1, &mut card2);
    }
    card1 as usize * (101 - card1 as usize) / 2 + card2 as usize - 1
}

#[inline]
pub fn index_to_card_pair(index: usize) -> (Card, Card) {
    let card1 = (103 - (103.0 * 103.0 - 8.0 * index as f64).sqrt().ceil() as u16) / 2;
    let card2 = index - card1 as usize * (101 - card1 as usize) / 2 + 1;
    (card1 as Card, card2 as Card)
}

pub const NUM_POSSIBLE_HANDS: usize = 52 * 51 / 2;

#[inline]
pub fn card_from_str(s: &str) -> Result<Card, String> {
    let s = s.trim();
    if s.len() != 2 {
        return Err(format!("invalid card '{}': expected 2 characters", s));
    }
    let rank = char_to_rank(s.chars().next().unwrap())?;
    let suit = char_to_suit(s.chars().nth(1).unwrap())?;
    Ok(4 * rank + suit)
}

#[inline]
pub fn flop_from_str(s: &str) -> Result<[Card; 3], String> {
    let s = s.trim();
    if s.len() != 4 && s.len() != 6 {
        return Err(format!("invalid flop '{}': expected 4 or 6 characters", s));
    }
    let cards: Vec<Card> = s.as_bytes().chunks(2)
        .map(|chunk| {
            let rank = char_to_rank(chunk[0] as char)?;
            let suit = char_to_suit(chunk[1] as char)?;
            Ok(4 * rank + suit)
        })
        .collect::<Result<Vec<Card>, String>>()?;
    if cards.len() != 3 {
        return Err(format!("invalid flop '{}': expected 3 cards", s));
    }
    Ok([cards[0], cards[1], cards[2]])
}

#[inline]
pub fn char_to_rank(c: char) -> Result<u8, String> {
    match c {
        'A' | 'a' => Ok(12),
        'K' | 'k' => Ok(11),
        'Q' | 'q' => Ok(10),
        'J' | 'j' => Ok(9),
        'T' | 't' => Ok(8),
        '2'..='9' => Ok(c as u8 - b'2'),
        _ => Err(format!("invalid rank character: '{}'", c)),
    }
}

#[inline]
pub fn char_to_suit(c: char) -> Result<u8, String> {
    match c {
        'c' | 'C' => Ok(0),
        'd' | 'D' => Ok(1),
        'h' | 'H' => Ok(2),
        's' | 'S' => Ok(3),
        _ => Err(format!("invalid suit character: '{}'", c)),
    }
}

#[inline]
pub fn rank_to_char(rank: u8) -> Result<char, String> {
    match rank {
        12 => Ok('A'),
        11 => Ok('K'),
        10 => Ok('Q'),
        9 => Ok('J'),
        8 => Ok('T'),
        0..=7 => Ok((rank + b'2') as char),
        _ => Err(format!("invalid rank: {}", rank)),
    }
}

#[inline]
pub fn suit_to_char(suit: u8) -> Result<char, String> {
    match suit {
        0 => Ok('c'),
        1 => Ok('d'),
        2 => Ok('h'),
        3 => Ok('s'),
        _ => Err(format!("invalid suit: {}", suit)),
    }
}

pub fn card_to_string(card: Card) -> Result<String, String> {
    if card >= 52 {
        return Err(format!("invalid card: {}", card));
    }
    let rank = card >> 2;
    let suit = card & 3;
    Ok(format!("{}{}", rank_to_char(rank)?, suit_to_char(suit)?))
}

pub fn hole_to_string(hole: (Card, Card)) -> Result<String, String> {
    let max_card = Card::max(hole.0, hole.1);
    let min_card = Card::min(hole.0, hole.1);
    Ok(format!("{}{}", card_to_string(max_card)?, card_to_string(min_card)?))
}

pub fn card_from_chars<T: Iterator<Item = char>>(chars: &mut T) -> Result<Card, String> {
    let rank_char = chars.next().ok_or_else(|| "unexpected end".to_string())?;
    let suit_char = chars.next().ok_or_else(|| "unexpected end".to_string())?;
    let rank = char_to_rank(rank_char)?;
    let suit = char_to_suit(suit_char)?;
    Ok((rank << 2) | suit)
}
