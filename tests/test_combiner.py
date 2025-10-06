from betcomb.providers.mock import MockStats, MockOdds
from betcomb.domain.combiner import best_double
from betcomb.config import SETTINGS

def test_best_double_mock_samples():
    stats = MockStats()
    odds = MockOdds()
    leagues = [l.code for l in stats.list_leagues()]
    matches = stats.fixtures(leagues, days=7)
    quotes = odds.odds_for_matches([m.id for m in matches])

    slip = best_double(
        matches=matches,
        quotes=quotes,
        p_cards_min=SETTINGS.p_cards_min,
        p_fhgoal_min=SETTINGS.p_fhgoal_min,
        target_total_odds=2.0,
    )
    assert slip is not None
    assert len(slip.picks) == 4
    assert slip.total_odds() >= 2.0
