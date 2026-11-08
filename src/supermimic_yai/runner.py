"""Live token runner: drives contact tokens from the env's current state with the actor (or the scripted token)."""
import numpy as np

from supermimic_yai.collect import _Recorder, _Stop
from supermimic_yai.npmlp import actor_forward, learned_kinds as _kinds
from supermimic_yai.scripted import TokenResult, capture_frame, run_token
from supermimic_yai.task import DROP_TIME, TOKEN_KINDS, TokenEpisode, budget_steps


def run_contact_token(env, tok, held, weights, action_scale):
    """-> (TokenResult, held). weights=None runs the scripted token through the same step path at zero noise."""
    # the scripted token judges its own grip; the analytic drop terminal is for learning
    ep = TokenEpisode.live(env, tok, held, budget_steps(env, tok), action_scale, np.inf if weights is None else DROP_TIME)
    ep.reset()
    why, res = [None], None
    if weights is None:
        def log(a, s2, r, done, info):
            why[0] = info.get("terminal_kind")
        try:
            res, _ = run_token(_Recorder(ep, 0.0, np.random.default_rng(0), log), tok, held)
        except _Stop:
            pass
    else:
        kind, info = TOKEN_KINDS.index(tok.kind), {}
        while "terminal_kind" not in info:
            _, _, _, info = ep.step(actor_forward(weights, ep.s, ep.goal, kind))
            capture_frame(env)
        why[0] = info["terminal_kind"]
    ok = why[0] == "success" or (res is not None and res.ok)
    reason = "" if ok else why[0] or res.reason
    return (TokenResult(ok, reason, ep.t * env.substeps * env.model.opt.timestep),
            tok.part if ok and tok.kind == "GRASP" else held)


class LearnedController:
    """Actor drives contact tokens whose kind is in learned_kinds; other contact tokens run scripted, all else run_token."""

    def __init__(self, weights, action_scale, learned_kinds=None):
        self.weights, self.scale, self.kinds, self.tags = weights, action_scale, _kinds(weights, learned_kinds), []

    def __call__(self, env, tok, held):
        if tok.kind not in TOKEN_KINDS:
            self.tags.append("scripted")
            return run_token(env, tok, held)
        w = self.weights if tok.kind in self.kinds else None
        self.tags.append("learned" if w is not None else "scripted")
        return run_contact_token(env, tok, held, w, self.scale)
