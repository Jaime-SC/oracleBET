from betcomb.domain.heuristics import cards_score, fh_over_0_5_score
from betcomb.domain.schemas import Match, League, Team
from datetime import datetime, timezone

def _m(**kw):
    base = dict(
        id="X",
        league=League(code="T", name="Test", region="eu"),
        date_utc=datetime.now(timezone.utc),
        home=Team(id="H", name="Home"),
        away=Team(id="A", name="Away"),
        home_cards_pg=2.0, away_cards_pg=2.0,
        referee_cards_pg=5.0,
        derby=False, knockout=False, intl_comp_bonus=0.0,
        home_fh_goals_pg=0.7, away_fh_goals_pg=0.6,
        home_shots_on_target_pg=4.0, away_shots_on_target_pg=3.5,
        ranking_diff=0.3, weather_penalty=0.0, congestion_penalty=0.0,
    )
    base.update(kw)
    return Match(**base)

def test_cards_score_increases_with_tension():
    m1 = _m(derby=False, knockout=False, intl_comp_bonus=0.0)
    m2 = _m(derby=True, knockout=True, intl_comp_bonus=0.3)
    s1 = cards_score(m1)
    s2 = cards_score(m2)
    assert s2.score > s1.score
    assert s2.prob > s1.prob

def test_fh_score_sensitive_to_shots_and_goals():
    a = _m(home_fh_goals_pg=0.4, away_fh_goals_pg=0.3, home_shots_on_target_pg=2.0, away_shots_on_target_pg=1.5)
    b = _m(home_fh_goals_pg=0.9, away_fh_goals_pg=0.8, home_shots_on_target_pg=6.0, away_shots_on_target_pg=5.5)
    sa = fh_over_0_5_score(a)
    sb = fh_over_0_5_score(b)
    assert sb.score > sa.score
    assert sb.prob > sa.prob
