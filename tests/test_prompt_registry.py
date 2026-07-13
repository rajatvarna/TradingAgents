"""Tests for ``tradingagents.audit.prompt_registry`` (T1.4).

Two layers:

1. **Registry contract** — load / render / cache / hash / version /
   missing-variable / path-traversal / trace_metadata semantics.

2. **Byte-identical equivalence** — rendering each migrated template
   with sample inputs must produce the exact string the original
   f-string in the agent factory would have produced. This is the
   hard proof that the migration is non-behavior-changing; without
   it the audit story is fiction.
"""

from __future__ import annotations

import hashlib

import pytest

from tradingagents.audit.prompt_registry import (
    DEFAULT_PROMPTS_DIR,
    PromptNotFoundError,
    PromptRegistry,
    default_registry,
    reset_default_registry,
)
from tradingagents.dataflows.config import set_config

# -------------------------------------------------------------------- #
# Registry contract
# -------------------------------------------------------------------- #


@pytest.mark.unit
class TestPromptRegistryLoad:
    def test_load_returns_text_and_hash(self, tmp_path):
        (tmp_path / "x.v1.txt").write_text("hello $name")
        r = PromptRegistry(base_dir=tmp_path)
        text, h = r.load("x", "v1")
        assert text == "hello $name"
        assert h == hashlib.sha256(b"hello $name").hexdigest()

    def test_load_respects_subdirs(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b.v1.txt").write_text("ok")
        r = PromptRegistry(base_dir=tmp_path)
        text, _ = r.load("a/b", "v1")
        assert text == "ok"

    def test_load_missing_raises_PromptNotFoundError(self, tmp_path):
        r = PromptRegistry(base_dir=tmp_path)
        with pytest.raises(PromptNotFoundError):
            r.load("nope", "v1")

    def test_load_caches_after_first_read(self, tmp_path):
        path = tmp_path / "x.v1.txt"
        path.write_text("first")
        r = PromptRegistry(base_dir=tmp_path)
        text1, h1 = r.load("x", "v1")
        # Mutate file on disk — cache must hold the original text
        path.write_text("second")
        text2, h2 = r.load("x", "v1")
        assert text1 == text2 == "first"
        assert h1 == h2

    def test_different_versions_have_different_hashes(self, tmp_path):
        (tmp_path / "x.v1.txt").write_text("alpha")
        (tmp_path / "x.v2.txt").write_text("beta")
        r = PromptRegistry(base_dir=tmp_path)
        _, h1 = r.load("x", "v1")
        _, h2 = r.load("x", "v2")
        assert h1 != h2

    def test_path_traversal_blocked(self, tmp_path):
        """``..`` segments must not escape the registry's base_dir."""
        r = PromptRegistry(base_dir=tmp_path)
        with pytest.raises(PromptNotFoundError):
            r.load("../etc/passwd", "v1")

    def test_hash_is_sha256_hex(self, tmp_path):
        (tmp_path / "x.v1.txt").write_text("content")
        r = PromptRegistry(base_dir=tmp_path)
        _, h = r.load("x", "v1")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


@pytest.mark.unit
class TestPromptRegistryRender:
    def test_render_substitutes_dollar_vars(self, tmp_path):
        (tmp_path / "x.v1.txt").write_text("hello $name, you are $age")
        r = PromptRegistry(base_dir=tmp_path)
        rendered, h = r.render("x", name="Alice", age="30")
        assert rendered == "hello Alice, you are 30"

    def test_render_substitutes_braced_vars(self, tmp_path):
        (tmp_path / "x.v1.txt").write_text("${greeting}, world")
        r = PromptRegistry(base_dir=tmp_path)
        rendered, _ = r.render("x", greeting="Hi")
        assert rendered == "Hi, world"

    def test_render_returns_template_hash_not_rendered_hash(self, tmp_path):
        """Hash is of the TEMPLATE, not the rendered output. Two different
        renders of the same template must return the same hash."""
        (tmp_path / "x.v1.txt").write_text("$x")
        r = PromptRegistry(base_dir=tmp_path)
        _, h_a = r.render("x", x="A")
        _, h_b = r.render("x", x="B")
        assert h_a == h_b

    def test_render_missing_var_raises_KeyError(self, tmp_path):
        """Silent empty substitution would let typos at the call site
        produce blank sections in production prompts. Fail loud."""
        (tmp_path / "x.v1.txt").write_text("$one and $two")
        r = PromptRegistry(base_dir=tmp_path)
        with pytest.raises(KeyError) as exc_info:
            r.render("x", one="hello")  # 'two' missing
        # Error mentions which variable + which key/version
        assert "two" in str(exc_info.value)
        assert "x" in str(exc_info.value)

    def test_render_strips_trailing_newline(self, tmp_path):
        """File-stored templates almost always have a trailing \\n from the
        editor. rstrip preserves byte-identical equivalence with f-strings."""
        # File ends with `$x\n` (trailing newline)
        (tmp_path / "x.v1.txt").write_text("$x\n")
        r = PromptRegistry(base_dir=tmp_path)
        rendered, _ = r.render("x", x="content")
        assert rendered == "content"

    def test_trace_metadata_shape(self, tmp_path):
        (tmp_path / "researchers" / "bull.v1.txt").parent.mkdir()
        (tmp_path / "researchers" / "bull.v1.txt").write_text("x")
        r = PromptRegistry(base_dir=tmp_path)
        md = r.trace_metadata("researchers/bull", "v1")
        assert md["prompt_key"] == "researchers/bull"
        assert md["prompt_version"] == "v1"
        assert md["prompt_hash"]
        assert len(md["prompt_hash"]) == 64


@pytest.mark.unit
class TestDefaultRegistrySingleton:
    def test_default_registry_returns_same_instance(self):
        reset_default_registry()
        a = default_registry()
        b = default_registry()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = default_registry()
        reset_default_registry()
        b = default_registry()
        assert a is not b

    def test_default_points_at_packaged_prompts(self):
        reset_default_registry()
        r = default_registry()
        # The default base_dir is tradingagents/prompts at the package root
        assert r.base_dir == DEFAULT_PROMPTS_DIR
        # And the bull_researcher template should be loadable from there
        text, h = r.load("researchers/bull_researcher", "v1")
        assert "Bull Analyst" in text


# -------------------------------------------------------------------- #
# Byte-identical equivalence: rendered template == original f-string
# -------------------------------------------------------------------- #


_BULL_INPUTS = {
    "target_label": "stock",
    "fundamentals_label": "Company fundamentals report",
    "market_research_report": "MARKET",
    "sentiment_report": "SENTIMENT",
    "news_report": "NEWS",
    "fundamentals_report": "FUND",
    "history": "HIST",
    "current_response": "BEAR_LAST",
    "scope_guard": "SCOPE",
    "esg_report": "ESG",
    "derivatives_report": "DERIV",
    "user_research_block": "USER_RES",
}


def _bull_legacy_fstring(language_instruction: str, **i) -> str:
    """The exact pre-T1.4 f-string body, kept here as the equivalence reference.

    Any change to this function must be matched by a change to
    ``researchers/bull_researcher.v1.txt`` and a hash bump in the
    on-disk template — otherwise the agent silently shifts behavior.
    Pinning this in the test suite is what enforces that.
    """
    return f"""You are a Bull Analyst advocating for investing in the {i['target_label']}. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
Market research report: {i['market_research_report']}
Social media sentiment report: {i['sentiment_report']}
Latest world affairs news: {i['news_report']}
{i['fundamentals_label']}: {i['fundamentals_report']}
Scope guard: {i['scope_guard']}
ESG report: {i['esg_report']}
Derivatives / options report: {i['derivatives_report']}
{i['user_research_block']}
Conversation history of the debate: {i['history']}
Last bear argument: {i['current_response']}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
{language_instruction}"""


@pytest.mark.unit
class TestByteIdenticalEquivalence:
    """Each migrated template renders byte-identical to its legacy f-string.

    These tests are the hard floor of the T1.4 contract: if any
    template drifts from its pre-migration form, the assertion catches
    it before bad audit data is produced.
    """

    def setup_method(self):
        reset_default_registry()
        # English (default) — language_instruction renders as empty
        set_config({"output_language": "English"})

    def test_bull_researcher_byte_identical_english(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "researchers/bull_researcher",
            language_instruction="",
            **_BULL_INPUTS,
        )
        expected = _bull_legacy_fstring(language_instruction="", **_BULL_INPUTS)
        assert rendered == expected

    def test_bull_researcher_byte_identical_chinese(self):
        registry = default_registry()
        lang_inst = " Write your entire response in Chinese."
        rendered, _ = registry.render(
            "researchers/bull_researcher",
            language_instruction=lang_inst,
            **_BULL_INPUTS,
        )
        expected = _bull_legacy_fstring(language_instruction=lang_inst, **_BULL_INPUTS)
        assert rendered == expected


# -------------------------------------------------------------------- #
# Integration: agent factories pass prompt provenance metadata
# -------------------------------------------------------------------- #


@pytest.mark.unit
class TestAgentFactoryRecordsPromptMetadata:
    """Each migrated agent factory must call llm.invoke(prompt, config={
    "metadata": {"prompt_key": ..., "prompt_version": ..., "prompt_hash": ...}})
    so the trace records prompt provenance without callback changes."""

    def setup_method(self):
        reset_default_registry()
        set_config({"output_language": "English"})

    def _state_for_researchers(self) -> dict:
        return {
            "investment_debate_state": {
                "history": "h", "bull_history": "bh", "bear_history": "bhr",
                "current_response": "cr", "count": 0,
            },
            "market_report": "m",
            "sentiment_report": "s",
            "news_report": "n",
            "fundamentals_report": "f",
            "asset_type": "stock",
            "company_of_interest": "AAPL",
        }

    def test_bull_researcher_passes_prompt_metadata(self):
        from unittest.mock import MagicMock

        from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="bull says go")
        node = create_bull_researcher(llm)
        node(self._state_for_researchers())

        # llm.invoke called once with (prompt, config={"metadata": {...}})
        assert llm.invoke.call_count == 1
        args, kwargs = llm.invoke.call_args
        md = kwargs["config"]["metadata"]
        assert md["prompt_key"] == "researchers/bull_researcher"
        assert md["prompt_version"] == "v2"
        assert len(md["prompt_hash"]) == 64

    def test_bear_researcher_passes_prompt_metadata(self):
        from unittest.mock import MagicMock

        from tradingagents.agents.researchers.bear_researcher import create_bear_researcher

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="bear says no")
        node = create_bear_researcher(llm)
        node(self._state_for_researchers())

        args, kwargs = llm.invoke.call_args
        md = kwargs["config"]["metadata"]
        assert md["prompt_key"] == "researchers/bear_researcher"
        assert md["prompt_version"] == "v2"
        assert len(md["prompt_hash"]) == 64

    def test_aggressive_debator_passes_prompt_metadata(self):
        from unittest.mock import MagicMock

        from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="aggressive says yes")
        state = {
            "risk_debate_state": {
                "history": "h", "aggressive_history": "ah",
                "conservative_history": "ch", "neutral_history": "nh",
                "current_aggressive_response": "", "current_conservative_response": "",
                "current_neutral_response": "", "count": 0,
            },
            "market_report": "m", "sentiment_report": "s",
            "news_report": "n", "fundamentals_report": "f",
            "trader_investment_plan": "BUY",
        }
        node = create_aggressive_debator(llm)
        node(state)

        args, kwargs = llm.invoke.call_args
        md = kwargs["config"]["metadata"]
        assert md["prompt_key"] == "risk/aggressive"
        assert md["prompt_version"] == "v1"


@pytest.mark.unit
class TestPromptVersionOverride:
    """Pinning a custom version via state['prompt_versions'] resolves the
    matching template file. This is the rollout mechanism for new prompts."""

    def test_custom_version_loads_alternate_template(self, tmp_path):
        # Build a registry pointing at tmp_path; seed v1 + v2
        (tmp_path / "researchers").mkdir()
        (tmp_path / "researchers" / "bull_researcher.v1.txt").write_text(
            "v1 says $target_label"
        )
        (tmp_path / "researchers" / "bull_researcher.v2.txt").write_text(
            "v2 says $target_label"
        )
        registry = PromptRegistry(base_dir=tmp_path)

        t1, _ = registry.render(
            "researchers/bull_researcher", version="v1", target_label="x"
        )
        t2, _ = registry.render(
            "researchers/bull_researcher", version="v2", target_label="x"
        )
        assert t1 != t2
        assert "v1" in t1
        assert "v2" in t2


@pytest.mark.unit
class TestConfiguredPromptVersionsReachState:
    """``default_config['prompt_versions']`` must be threaded into agent
    state by ``TradingAgentsGraph``, or every registry-backed agent's
    ``state.get("prompt_versions", {})`` call returns ``{}`` and silently
    falls back to its hardcoded "v1" default — meaning the versions declared
    in config (v2 researchers, v2 portfolio manager, v3 trader) never take
    effect. Regression test for that bug."""

    def _build_graph(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from tradingagents.graph import trading_graph as tg

        monkeypatch.setattr(
            tg, "create_llm_client",
            lambda **kw: MagicMock(get_llm=lambda: MagicMock()),
        )
        monkeypatch.setattr(
            tg.GraphSetup, "setup_graph",
            lambda self, selected_analysts=None, **kw: MagicMock(),
        )

        cfg = dict(tg.DEFAULT_CONFIG)
        cfg["audit_dir"] = str(tmp_path)
        cfg["audit_full_trace_enabled"] = False
        cfg["checkpoint_enabled"] = False

        return tg.TradingAgentsGraph(config=cfg), cfg

    def test_propagate_injects_configured_prompt_versions(self, tmp_path, monkeypatch):
        ta, cfg = self._build_graph(tmp_path, monkeypatch)

        captured = {}

        class _ProbeError(Exception):
            pass

        def _capture_invoke(state, **kwargs):
            captured["state"] = state
            raise _ProbeError()

        ta.graph.invoke = _capture_invoke

        with pytest.raises(_ProbeError):
            ta.propagate("AAPL", "2026-01-02")

        assert captured["state"]["prompt_versions"] == cfg["prompt_versions"]
        # Sanity: the configured versions are the fork's improved templates,
        # not the bare "v1" every agent falls back to when this key is missing.
        assert captured["state"]["prompt_versions"]["trader/trader_system"] == "v3"
        assert captured["state"]["prompt_versions"]["researchers/bull_researcher"] == "v2"

    def test_stream_run_injects_configured_prompt_versions(self, tmp_path, monkeypatch):
        ta, cfg = self._build_graph(tmp_path, monkeypatch)

        captured = {}

        class _ProbeError(Exception):
            pass

        def _capture_stream(state, **kwargs):
            captured["state"] = state
            raise _ProbeError()

        ta.graph.stream = _capture_stream

        with pytest.raises(_ProbeError):
            list(ta.stream_run("AAPL", "2026-01-02"))

        assert captured["state"]["prompt_versions"] == cfg["prompt_versions"]


# -------------------------------------------------------------------- #
# render_with_shared (A1) — shared partial composition
# -------------------------------------------------------------------- #


@pytest.mark.unit
class TestRenderWithShared:
    def test_composes_shared_partial_into_main_template(self, tmp_path):
        (tmp_path / "_shared").mkdir()
        (tmp_path / "_shared" / "greeting.v1.txt").write_text("Hello, $name!")
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "demo.v1.txt").write_text("${greeting_block}\nBody text.")

        registry = PromptRegistry(base_dir=tmp_path)
        rendered, digest, shared_hashes = registry.render_with_shared(
            "agents/demo",
            shared={"greeting_block": ("_shared/greeting", "v1")},
            name="World",
        )

        assert rendered == "Hello, World!\nBody text."
        assert len(digest) == 64
        assert set(shared_hashes) == {"_shared/greeting"}
        assert len(shared_hashes["_shared/greeting"]) == 64

    def test_shared_hash_is_the_shared_templates_own_hash(self, tmp_path):
        (tmp_path / "_shared").mkdir()
        (tmp_path / "_shared" / "block.v1.txt").write_text("SHARED TEXT")
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "demo.v1.txt").write_text("${block}")

        registry = PromptRegistry(base_dir=tmp_path)
        _, _, shared_hashes = registry.render_with_shared(
            "agents/demo", shared={"block": ("_shared/block", "v1")}
        )
        _, expected_hash = registry.load("_shared/block", "v1")
        assert shared_hashes["_shared/block"] == expected_hash

    def test_multiple_shared_partials_compose_independently(self, tmp_path):
        (tmp_path / "_shared").mkdir()
        (tmp_path / "_shared" / "a.v1.txt").write_text("BLOCK_A")
        (tmp_path / "_shared" / "b.v1.txt").write_text("BLOCK_B")
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "demo.v1.txt").write_text("${a_block}|${b_block}")

        registry = PromptRegistry(base_dir=tmp_path)
        rendered, _, shared_hashes = registry.render_with_shared(
            "agents/demo",
            shared={"a_block": ("_shared/a", "v1"), "b_block": ("_shared/b", "v1")},
        )
        assert rendered == "BLOCK_A|BLOCK_B"
        assert set(shared_hashes) == {"_shared/a", "_shared/b"}

    def test_no_shared_partials_behaves_like_plain_render(self, tmp_path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "demo.v1.txt").write_text("plain $name")

        registry = PromptRegistry(base_dir=tmp_path)
        rendered, digest, shared_hashes = registry.render_with_shared("agents/demo", name="X")
        plain_rendered, plain_digest = registry.render("agents/demo", name="X")

        assert rendered == plain_rendered == "plain X"
        assert digest == plain_digest
        assert shared_hashes == {}

    def test_shared_partial_version_is_independently_selectable(self, tmp_path):
        (tmp_path / "_shared").mkdir()
        (tmp_path / "_shared" / "block.v1.txt").write_text("v1 text")
        (tmp_path / "_shared" / "block.v2.txt").write_text("v2 text")
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "demo.v1.txt").write_text("${block}")

        registry = PromptRegistry(base_dir=tmp_path)
        rendered_v1, _, _ = registry.render_with_shared(
            "agents/demo", shared={"block": ("_shared/block", "v1")}
        )
        rendered_v2, _, _ = registry.render_with_shared(
            "agents/demo", shared={"block": ("_shared/block", "v2")}
        )
        assert rendered_v1 == "v1 text"
        assert rendered_v2 == "v2 text"


@pytest.mark.unit
class TestSharedPartialsShipped:
    """The two A1 shared partials exist, are non-empty, and read as intended."""

    def test_data_integrity_partial_loads(self):
        text, digest = default_registry().load("_shared/data_integrity", "v1")
        assert "never invent" in text.lower()
        assert len(digest) == 64

    def test_calibration_partial_loads(self):
        text, digest = default_registry().load("_shared/calibration", "v1")
        assert "confidence" in text.lower()
        assert "falsifier" in text.lower() or "change your conclusion" in text.lower()
        assert len(digest) == 64


# -------------------------------------------------------------------- #
# Config <-> registry consistency (A6 "verify" guard)
#
# Cheap regression guard against exactly the failure mode T1.4's docstring
# warns about: default_config.py declaring a version that has no matching
# file on disk (a typo, or a version bumped in one place and not the
# other). Every key in default_config["prompt_versions"] must resolve.
# -------------------------------------------------------------------- #


@pytest.mark.unit
class TestPromptVersionsConfigResolves:
    def test_every_configured_prompt_version_has_a_template_file(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        registry = default_registry()
        prompt_versions = DEFAULT_CONFIG["prompt_versions"]
        assert prompt_versions, "default_config.py's prompt_versions must not be empty"

        missing = []
        for key, version in prompt_versions.items():
            try:
                registry.load(key, version)
            except PromptNotFoundError:
                missing.append((key, version))

        assert not missing, f"prompt_versions declares versions with no template file: {missing}"


# -------------------------------------------------------------------- #
# A3 debate-layer redesign: researchers v3 (evidence over rhetoric)
#
# v3 is new content, not a byte-identical migration of anything, so these
# are golden-content checks (does the rendered template carry the
# structural markers the A3 redesign requires) plus node-level tests that
# selecting v3 via prompt_versions actually renders and records shared
# partial hashes. v3 is shipped available-but-not-default: default_config
# still selects v2, per the A6 merge-gate convention (no default flips
# without a scorecard) documented in docs/PROMPT_AND_CORE_FEATURES_PLAN.md.
# -------------------------------------------------------------------- #


@pytest.mark.unit
class TestResearcherV3Content:
    def _render(self, key):
        registry = default_registry()
        rendered, _, shared_hashes = registry.render_with_shared(
            key,
            version="v3",
            shared={
                "data_integrity_block": ("_shared/data_integrity", "v1"),
                "calibration_block": ("_shared/calibration", "v1"),
            },
            **_BULL_INPUTS,
            group_sector_report="GROUP",
            market_phase_report="PHASE",
            monster_stock_block="MONSTER",
            language_instruction="",
        )
        return rendered, shared_hashes

    def test_bull_v3_carries_evidence_over_rhetoric_markers(self):
        rendered, shared_hashes = self._render("researchers/bull_researcher")
        assert "steelman" in rendered.lower()
        assert "P(bull thesis plays out" in rendered
        assert "falsifiers" in rendered.lower() or "flip you to" in rendered.lower()
        # Shared data-integrity block actually composed in, not just referenced
        assert "never invent" in rendered.lower()
        assert set(shared_hashes) == {"_shared/data_integrity", "_shared/calibration"}

    def test_bear_v3_carries_evidence_over_rhetoric_markers(self):
        rendered, shared_hashes = self._render("researchers/bear_researcher")
        assert "steelman" in rendered.lower()
        assert "P(bear thesis plays out" in rendered
        assert "falsifiers" in rendered.lower() or "flip you to" in rendered.lower()
        assert "never invent" in rendered.lower()

    def test_v3_drops_conversational_style_instruction(self):
        """The A3 rewrite's whole point is dropping the rhetoric-optimizing
        'conversational style, debating effectively' framing v1/v2 use."""
        rendered, _ = self._render("researchers/bull_researcher")
        assert "conversational style" not in rendered.lower()

    def test_no_unresolved_placeholders(self):
        rendered, _ = self._render("researchers/bull_researcher")
        # A leftover ${...} means a variable was referenced but not supplied
        assert "${" not in rendered


@pytest.mark.unit
class TestResearcherDefaultVersionUnchangedByV3Availability:
    """Shipping v3 must not silently change what runs by default — that
    only happens via an explicit default_config bump backed by an A6
    scorecard."""

    def test_default_config_still_selects_v2(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["prompt_versions"]["researchers/bull_researcher"] == "v2"
        assert DEFAULT_CONFIG["prompt_versions"]["researchers/bear_researcher"] == "v2"

    def test_node_without_explicit_override_still_renders_v2(self):
        from unittest.mock import MagicMock

        from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="bull says go")
        node = create_bull_researcher(llm)
        state = TestAgentFactoryRecordsPromptMetadata()._state_for_researchers()
        node(state)

        _, kwargs = llm.invoke.call_args
        assert kwargs["config"]["metadata"]["prompt_version"] == "v2"

    def test_node_honors_explicit_v3_override(self):
        from unittest.mock import MagicMock

        from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="bull says go")
        node = create_bull_researcher(llm)
        state = TestAgentFactoryRecordsPromptMetadata()._state_for_researchers()
        state["prompt_versions"] = {"researchers/bull_researcher": "v3"}
        node(state)

        _, kwargs = llm.invoke.call_args
        md = kwargs["config"]["metadata"]
        assert md["prompt_version"] == "v3"
        assert set(md["shared_prompt_hashes"]) == {"_shared/data_integrity", "_shared/calibration"}


# -------------------------------------------------------------------- #
# A3 debate-layer redesign: risk team v2 (personas -> risk functions)
# -------------------------------------------------------------------- #

_RISK_INPUTS = {
    "trader_decision": "BUY, entry 100, stop 92, target 120",
    "market_research_report": "MARKET",
    "sentiment_report": "SENTIMENT",
    "news_report": "NEWS",
    "fundamentals_report": "FUND",
    "scope_guard": "SCOPE",
    "history": "HIST",
}
_RISK_SHARED = {
    "data_integrity_block": ("_shared/data_integrity", "v1"),
    "calibration_block": ("_shared/calibration", "v1"),
}


@pytest.mark.unit
class TestRiskTeamV2Content:
    def test_aggressive_v2_reframed_as_upside_opportunity_cost_analyst(self):
        registry = default_registry()
        rendered, _, shared_hashes = registry.render_with_shared(
            "risk/aggressive",
            version="v2",
            shared=_RISK_SHARED,
            **_RISK_INPUTS,
            current_conservative_response="CONS",
            current_neutral_response="NEUT",
            language_instruction="",
        )
        assert "Upside / Opportunity-Cost Analyst" in rendered
        assert "opportunity cost" in rendered.lower()
        assert "double-counting" in rendered.lower()
        assert "Case | Probability | Price impact | Portfolio impact" in rendered
        assert "actively champion" not in rendered.lower()
        assert "never invent" in rendered.lower()
        assert set(shared_hashes) == {"_shared/data_integrity", "_shared/calibration"}

    def test_conservative_v2_reframed_as_downside_analyst(self):
        registry = default_registry()
        rendered, _, _ = registry.render_with_shared(
            "risk/conservative",
            version="v2",
            shared=_RISK_SHARED,
            **_RISK_INPUTS,
            current_aggressive_response="AGG",
            current_neutral_response="NEUT",
            language_instruction="",
        )
        assert "Downside Analyst" in rendered
        assert "kelly" in rendered.lower()
        assert "stop-loss stress test" in rendered.lower()
        assert "Case | Probability | Price impact | Portfolio impact" in rendered

    def test_neutral_v2_reframed_as_calibration_referee(self):
        registry = default_registry()
        rendered, _, _ = registry.render_with_shared(
            "risk/neutral",
            version="v2",
            shared=_RISK_SHARED,
            **_RISK_INPUTS,
            current_aggressive_response="AGG",
            current_conservative_response="CONS",
            language_instruction="",
        )
        assert "Calibration Referee" in rendered
        assert "factual disputes" in rendered.lower()
        assert "judgment disagreements" in rendered.lower()
        assert "reconciled" in rendered.lower()

    def test_v2_drops_no_special_formatting_instruction(self):
        """v1's 'output conversationally ... no special formatting' fights
        the scenario-table requirement v2 replaces it with."""
        registry = default_registry()
        rendered, _, _ = registry.render_with_shared(
            "risk/aggressive",
            version="v2",
            shared=_RISK_SHARED,
            **_RISK_INPUTS,
            current_conservative_response="",
            current_neutral_response="",
            language_instruction="",
        )
        assert "no special formatting" not in rendered.lower()

    def test_default_config_still_selects_v1_for_risk_team(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["prompt_versions"]["risk/aggressive"] == "v1"
        assert DEFAULT_CONFIG["prompt_versions"]["risk/conservative"] == "v1"
        assert DEFAULT_CONFIG["prompt_versions"]["risk/neutral"] == "v1"

    def test_aggressive_node_honors_explicit_v2_override(self):
        from unittest.mock import MagicMock

        from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="upside view")
        node = create_aggressive_debator(llm)
        state = {
            "risk_debate_state": {
                "history": "h", "aggressive_history": "ah",
                "conservative_history": "ch", "neutral_history": "nh",
                "current_aggressive_response": "", "current_conservative_response": "",
                "current_neutral_response": "", "count": 0,
            },
            "market_report": "m", "sentiment_report": "s",
            "news_report": "n", "fundamentals_report": "f",
            "trader_investment_plan": "BUY",
            "prompt_versions": {"risk/aggressive": "v2"},
        }
        node(state)

        _, kwargs = llm.invoke.call_args
        md = kwargs["config"]["metadata"]
        assert md["prompt_version"] == "v2"
        assert set(md["shared_prompt_hashes"]) == {"_shared/data_integrity", "_shared/calibration"}
