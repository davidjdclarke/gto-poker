from dataclasses import dataclass
import random

SUITS = ["h", "d", "c", "s"]
SUIT_NAMES = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades"}
RANK_NAMES = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
              9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
RANK_FROM_CHAR = {v: k for k, v in RANK_NAMES.items()}


@dataclass(frozen=True)
class Card:
    rank: int  # 2-14 (14=Ace)
    suit: str  # h, d, c, s

    def __str__(self):
        return f"{RANK_NAMES[self.rank]}{self.suit}"

    def to_dict(self):
        return str(self)


class Deck:
    def __init__(self):
        self.cards: list[Card] = []
        self.reset()

    def reset(self):
        self.cards = [Card(rank, suit) for suit in SUITS for rank in range(2, 15)]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> list[Card]:
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def burn(self):
        if self.cards:
            self.cards.pop(0)
