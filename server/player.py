from dataclasses import dataclass, field
from server.deck import Card


@dataclass
class Player:
    id: str
    name: str
    chips: int
    seat: int
    is_human: bool = False
    hole_cards: list[Card] = field(default_factory=list)
    is_folded: bool = False
    is_all_in: bool = False
    current_bet: int = 0
    has_acted: bool = False
    ai_style: str = "balanced"  # tight, loose, aggressive, balanced

    def reset_for_hand(self):
        self.hole_cards = []
        self.is_folded = False
        self.is_all_in = False
        self.current_bet = 0
        self.has_acted = False

    def place_bet(self, amount: int) -> int:
        actual = min(amount, self.chips)
        self.chips -= actual
        self.current_bet += actual
        if self.chips == 0:
            self.is_all_in = True
        return actual

    def to_dict(self, reveal: bool = False, is_self: bool = False) -> dict:
        cards = None
        if is_self or reveal:
            cards = [str(c) for c in self.hole_cards] if self.hole_cards else []
        elif self.hole_cards and not self.is_folded:
            cards = ["back", "back"]

        return {
            "id": self.id,
            "name": self.name,
            "chips": self.chips,
            "seat": self.seat,
            "current_bet": self.current_bet,
            "is_folded": self.is_folded,
            "is_all_in": self.is_all_in,
            "is_human": self.is_human,
            "hole_cards": cards,
            "ai_style": self.ai_style if not self.is_human else None,
        }
