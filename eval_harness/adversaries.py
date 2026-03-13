"""
Adversary suite: a battery of exploit bots to stress-test GTO strategy.

Each bot targets a specific weakness. Measuring bb/100 against each reveals
which exploits the strategy is vulnerable to.

Bots:
    NitBot         - extremely tight preflop, only plays premiums
    AggroBot       - hyper-aggressive 3-bettor, always raising
    OverfoldBot    - folds to any postflop bet (cbet exploitable)
    CallStationBot - never folds postflop, calls everything
    DonkBot        - leads into raiser with weird sizings
    WeirdSizingBot - uses off-tree bet sizes (75% pot, min-clicks, overbets)
    PerturbBot     - plays random off-tree actions with small probability
"""
import random
import math
from server.gto.equity import hand_strength_bucket, hand_equity
from server.gto.abstraction import classify_hand_type
from eval_harness.match_engine import Agent, AgentDecision, HandContext
from eval_harness.fast_equity import fast_bot_equity


def _fast_equity(ctx: HandContext) -> float:
    """Fast equity estimate using cache for preflop, few sims for postflop."""
    try:
        return fast_bot_equity(ctx.hole_cards, ctx.community_cards)
    except Exception:
        return 0.5


class NitBot(Agent):
    """Extremely tight preflop. Only plays top ~15% of hands.
    Postflop plays straightforward: bet strong, check/fold weak."""

    name = "NitBot"

    def decide(self, ctx: HandContext) -> AgentDecision:
        to_call = ctx.current_bet - ctx.my_bet

        if ctx.phase == "preflop":
            equity = _fast_equity(ctx)
            if equity > 0.65:
                # Premium: raise
                raise_to = ctx.current_bet + max(ctx.min_raise, int(ctx.big_blind * 3))
                raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                return AgentDecision("raise", raise_to)
            elif equity > 0.52:
                # Decent: call if cheap
                if to_call <= ctx.big_blind * 2:
                    return AgentDecision("call" if to_call > 0 else "check")
                return AgentDecision("fold") if to_call > 0 else AgentDecision("check")
            else:
                return AgentDecision("fold") if to_call > 0 else AgentDecision("check")
        else:
            # Postflop: straightforward
            equity = _fast_equity(ctx)
            if equity > 0.7:
                # Strong: bet 2/3 pot
                bet_size = max(ctx.min_raise, int(ctx.pot * 0.67))
                raise_to = ctx.current_bet + bet_size
                raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                if raise_to > ctx.current_bet:
                    return AgentDecision("raise", raise_to)
                return AgentDecision("call" if to_call > 0 else "check")
            elif equity > 0.45:
                return AgentDecision("call" if to_call > 0 else "check")
            else:
                return AgentDecision("fold") if to_call > 0 else AgentDecision("check")


class AggroBot(Agent):
    """Hyper-aggressive: 3-bets preflop with wide range, barrels postflop."""

    name = "AggroBot"

    def decide(self, ctx: HandContext) -> AgentDecision:
        to_call = ctx.current_bet - ctx.my_bet

        if ctx.phase == "preflop":
            # 3-bet ~50% of hands facing a raise
            if to_call > ctx.big_blind:
                if random.random() < 0.50:
                    raise_to = max(ctx.current_bet * 3, ctx.current_bet + ctx.min_raise)
                    raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                    return AgentDecision("raise", raise_to)
                elif random.random() < 0.3:
                    return AgentDecision("call")
                else:
                    return AgentDecision("fold")
            else:
                # Open raise ~70% of hands
                if random.random() < 0.70:
                    raise_to = ctx.current_bet + max(ctx.min_raise, int(ctx.big_blind * 2.5))
                    raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                    return AgentDecision("raise", raise_to)
                return AgentDecision("call" if to_call > 0 else "check")
        else:
            # Postflop: barrel ~65% of the time
            if to_call > 0:
                # Facing a bet: raise sometimes, call often
                if random.random() < 0.25:
                    raise_to = ctx.current_bet + max(ctx.min_raise, int(ctx.pot * 0.75))
                    raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                    if raise_to > ctx.current_bet:
                        return AgentDecision("raise", raise_to)
                if random.random() < 0.6:
                    return AgentDecision("call")
                return AgentDecision("fold")
            else:
                # No bet: lead ~65%
                if random.random() < 0.65:
                    bet_size = max(ctx.min_raise, int(ctx.pot * 0.67))
                    raise_to = bet_size
                    raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                    if raise_to > 0:
                        return AgentDecision("raise", raise_to)
                return AgentDecision("check")


class OverfoldBot(Agent):
    """Folds to any postflop bet. Plays reasonably preflop.
    Exploitable by anyone who cbets at any frequency."""

    name = "OverfoldBot"

    def decide(self, ctx: HandContext) -> AgentDecision:
        to_call = ctx.current_bet - ctx.my_bet

        if ctx.phase == "preflop":
            equity = _fast_equity(ctx)
            if equity > 0.55:
                raise_to = ctx.current_bet + max(ctx.min_raise, int(ctx.big_blind * 2.5))
                raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                return AgentDecision("raise", raise_to)
            elif equity > 0.40:
                return AgentDecision("call" if to_call > 0 else "check")
            else:
                return AgentDecision("fold") if to_call > 0 else AgentDecision("check")
        else:
            # Postflop: fold to any bet, check back everything
            if to_call > 0:
                return AgentDecision("fold")
            return AgentDecision("check")


class CallStationBot(Agent):
    """Never folds postflop. Calls everything. Barely raises."""

    name = "CallStationBot"

    def decide(self, ctx: HandContext) -> AgentDecision:
        to_call = ctx.current_bet - ctx.my_bet

        if ctx.phase == "preflop":
            # Call almost everything preflop too
            if to_call > ctx.big_blind * 5:
                equity = _fast_equity(ctx)
                if equity < 0.35:
                    return AgentDecision("fold")
            return AgentDecision("call" if to_call > 0 else "check")
        else:
            # Postflop: call everything, occasionally bet the nuts
            equity = _fast_equity(ctx)
            if to_call > 0:
                return AgentDecision("call")
            elif equity > 0.85:
                # Bet when very strong
                bet_size = max(ctx.min_raise, int(ctx.pot * 0.5))
                raise_to = bet_size
                raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                if raise_to > 0:
                    return AgentDecision("raise", raise_to)
            return AgentDecision("check")


class DonkBot(Agent):
    """Leads into the preflop raiser with random sizings.
    Tests how the GTO strategy handles donk bets (off-tree for many solvers)."""

    name = "DonkBot"

    def decide(self, ctx: HandContext) -> AgentDecision:
        to_call = ctx.current_bet - ctx.my_bet

        if ctx.phase == "preflop":
            equity = _fast_equity(ctx)
            if equity > 0.48:
                return AgentDecision("call" if to_call > 0 else "check")
            return AgentDecision("fold") if to_call > 0 else AgentDecision("check")
        else:
            # Donk: lead into raiser with varied sizings
            if to_call > 0:
                # Facing a bet: call mostly
                return AgentDecision("call")

            # Lead out ~55% of the time with random sizes
            if random.random() < 0.55:
                # Random sizing: 20%, 40%, 75%, 120% pot
                pct = random.choice([0.20, 0.40, 0.75, 1.2])
                bet_size = max(ctx.min_raise, int(ctx.pot * pct))
                raise_to = bet_size
                raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                if raise_to > 0:
                    return AgentDecision("raise", raise_to)
            return AgentDecision("check")


class WeirdSizingBot(Agent):
    """Uses bet sizes that don't map cleanly into the abstraction.
    Tests action translation robustness."""

    name = "WeirdSizingBot"

    # Off-tree sizings
    PREFLOP_SIZES = [2.0, 2.2, 3.5, 5.0]  # BB multipliers for open
    POSTFLOP_PCTS = [0.15, 0.28, 0.42, 0.75, 1.3, 2.0]  # pot fractions

    def decide(self, ctx: HandContext) -> AgentDecision:
        to_call = ctx.current_bet - ctx.my_bet

        if ctx.phase == "preflop":
            equity = _fast_equity(ctx)
            if equity > 0.50:
                # Open with weird size
                mult = random.choice(self.PREFLOP_SIZES)
                raise_to = int(ctx.big_blind * mult) + ctx.current_bet
                raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                if raise_to > ctx.current_bet:
                    return AgentDecision("raise", raise_to)
                return AgentDecision("call" if to_call > 0 else "check")
            elif equity > 0.38:
                return AgentDecision("call" if to_call > 0 else "check")
            return AgentDecision("fold") if to_call > 0 else AgentDecision("check")
        else:
            equity = _fast_equity(ctx)
            if to_call > 0:
                if equity > 0.4:
                    if random.random() < 0.3:
                        pct = random.choice(self.POSTFLOP_PCTS)
                        raise_to = ctx.current_bet + max(ctx.min_raise, int(ctx.pot * pct))
                        raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                        if raise_to > ctx.current_bet:
                            return AgentDecision("raise", raise_to)
                    return AgentDecision("call")
                return AgentDecision("fold")
            else:
                if equity > 0.35:
                    pct = random.choice(self.POSTFLOP_PCTS)
                    bet_size = max(ctx.min_raise, int(ctx.pot * pct))
                    raise_to = bet_size
                    raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                    if raise_to > 0:
                        return AgentDecision("raise", raise_to)
                return AgentDecision("check")


class PerturbBot(Agent):
    """Plays like GTO but randomly deviates with off-tree actions.
    Tests strategy robustness to small perturbations."""

    name = "PerturbBot"

    def __init__(self, trainer, perturb_rate: float = 0.15):
        self.trainer = trainer
        self.gto_agent = None  # Lazy init
        self.perturb_rate = perturb_rate

    def _get_gto(self):
        if self.gto_agent is None:
            from eval_harness.match_engine import GTOAgent
            self.gto_agent = GTOAgent(self.trainer, name="PerturbGTO")
        return self.gto_agent

    def decide(self, ctx: HandContext) -> AgentDecision:
        if random.random() < self.perturb_rate:
            return self._random_action(ctx)
        return self._get_gto().decide(ctx)

    def _random_action(self, ctx: HandContext) -> AgentDecision:
        to_call = ctx.current_bet - ctx.my_bet
        choice = random.random()

        if choice < 0.15:
            return AgentDecision("fold") if to_call > 0 else AgentDecision("check")
        elif choice < 0.45:
            return AgentDecision("call" if to_call > 0 else "check")
        else:
            # Random raise with random sizing
            pct = random.choice([0.25, 0.42, 0.67, 0.85, 1.1, 1.5])
            raise_to = ctx.current_bet + max(ctx.min_raise, int(ctx.pot * pct))
            raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
            if raise_to > ctx.current_bet:
                return AgentDecision("raise", raise_to)
            return AgentDecision("call" if to_call > 0 else "check")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def get_all_adversaries(trainer=None):
    """Return list of all adversary instances."""
    bots = [
        NitBot(),
        AggroBot(),
        OverfoldBot(),
        CallStationBot(),
        DonkBot(),
        WeirdSizingBot(),
    ]
    if trainer is not None:
        bots.append(PerturbBot(trainer))
    return bots
