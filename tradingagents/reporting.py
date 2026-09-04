"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

from datetime import datetime
from pathlib import Path


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if final_state.get("options_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "options.md").write_text(final_state["options_report"], encoding="utf-8")
        analyst_parts.append(("Options Analyst", final_state["options_report"]))
    if final_state.get("esg_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "esg.md").write_text(final_state["esg_report"], encoding="utf-8")
        analyst_parts.append(("ESG Analyst", final_state["esg_report"]))
    if final_state.get("derivatives_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "derivatives.md").write_text(final_state["derivatives_report"], encoding="utf-8")
        analyst_parts.append(("Derivatives Analyst", final_state["derivatives_report"]))
    if final_state.get("technical_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "technical.md").write_text(final_state["technical_report"], encoding="utf-8")
        analyst_parts.append(("Technical Analyst", final_state["technical_report"]))
    if final_state.get("quant_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "quant.md").write_text(final_state["quant_report"], encoding="utf-8")
        analyst_parts.append(("Quant Analyst", final_state["quant_report"]))
    if final_state.get("alternative_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "alternative.md").write_text(final_state["alternative_report"], encoding="utf-8")
        analyst_parts.append(("Alternative Data Analyst", final_state["alternative_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

    # 5. Portfolio Manager — standalone or via risk debate (PR #1293)
    risk_state = final_state.get("risk_debate_state") or {}
    pm_state = final_state.get("portfolio_manager_state") or {}
    portfolio_decision = (
        risk_state.get("judge_decision")
        or pm_state.get("judge_decision")
        or final_state.get("portfolio_decision")
    )
    if portfolio_decision:
        portfolio_dir = save_path / "5_portfolio"
        portfolio_dir.mkdir(exist_ok=True)
        (portfolio_dir / "decision.md").write_text(portfolio_decision, encoding="utf-8")
        sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{portfolio_decision}")

    # 5b. Data Sources / Provenance (PR #1270, #1197)
    data_sources = final_state.get("data_sources") or final_state.get("data_provenance")
    if data_sources:
        if isinstance(data_sources, list):
            sources_text = "\n".join(f"- {s}" if not str(s).startswith("- ") else str(s) for s in data_sources)
        elif isinstance(data_sources, dict):
            sources_text = "\n".join(f"- **{k}**: {v}" for k, v in data_sources.items())
        else:
            sources_text = str(data_sources)
        (save_path / "data_sources.md").write_text(sources_text, encoding="utf-8")
        sections.append(f"## Data Sources\n\n{sources_text}")

    # 6. Evidence Audit (PR #1105)
    evidence_audit = build_evidence_audit(final_state)
    if evidence_audit["has_content"]:
        import json as _json
        evidence_dir = save_path / "6_evidence"
        evidence_dir.mkdir(exist_ok=True)
        (evidence_dir / "audit.md").write_text(evidence_audit["markdown"], encoding="utf-8")
        (evidence_dir / "evidence_audit.json").write_text(
            _json.dumps(evidence_audit["data"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        sections.append(f"## VI. Evidence Audit\n\n{evidence_audit['markdown']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\n"
    # Include trade_date for auditing/backtesting reproducibility (#1270, #1197)
    td = final_state.get("trade_date") or final_state.get("tradeDate")
    if td:
        header += f"Trade Date: {td}\n"
    header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_text = header + "\n\n".join(sections)
    try:
        from cli.report_headings import transform as _prune_report_headings
        report_text = _prune_report_headings(report_text)
    except ImportError:
        pass
    (save_path / "complete_report.md").write_text(report_text, encoding="utf-8")
    return save_path / "complete_report.md"


def build_evidence_audit(final_state: dict) -> dict:
    """Compile evidence audit information from final_state into structured data.

    Returns a dict with:
    - ``has_content``: True if there is any evidence audit data worth writing.
    - ``markdown``: A markdown-formatted audit summary string.
    - ``data``: A JSON-serialisable dict of all audit fields.
    """
    decision_status = final_state.get("evidence_decision_status", "actionable")
    evidence_actionable = final_state.get("evidence_actionable", True)
    evidence_summary = final_state.get("evidence_summary", "")
    evidence_warnings = final_state.get("evidence_warnings") or []
    blocking_reasons = final_state.get("evidence_blocking_reasons") or []
    math_events = final_state.get("math_guardrail_events") or []
    citation_result = final_state.get("citation_verification") or {}
    ledger = final_state.get("evidence_ledger") or {}
    anchors = final_state.get("quantitative_anchors") or []

    has_content = bool(
        evidence_summary
        or evidence_warnings
        or blocking_reasons
        or math_events
        or citation_result
        or (ledger.get("items") if isinstance(ledger, dict) else False)
    )

    lines = [
        f"**Evidence Decision Status**: {decision_status}",
        f"**Actionable**: {evidence_actionable}",
    ]

    if evidence_summary:
        lines += ["", "**Evidence Summary:**", evidence_summary]

    if anchors:
        lines += ["", "**Quantitative Anchors:**"]
        for anchor in anchors:
            if isinstance(anchor, dict):
                sym = anchor.get("symbol", "?")
                price = anchor.get("current_price", "?")
                ev_id = anchor.get("evidence_id", "")
                lines.append(f"- {sym}: current_price={price}  [{ev_id}]")

    if math_events:
        lines += ["", "**Math Guardrail Events:**"]
        for evt in math_events:
            if isinstance(evt, dict):
                action = evt.get("action", "?")
                reason = evt.get("reason", "?")
                lines.append(f"- [{action.upper()}] {reason}")

    if citation_result:
        verified = citation_result.get("verified_ids", [])
        missing = citation_result.get("missing_ids", [])
        lines += ["", "**Citation Verification:**"]
        if verified:
            lines.append(f"- Verified IDs: {', '.join(verified)}")
        if missing:
            lines.append(f"- Unresolved IDs: {', '.join(missing)}")

    if blocking_reasons:
        lines += ["", "**Blocking Reasons:**"]
        for reason in blocking_reasons:
            lines.append(f"- {reason}")

    if evidence_warnings:
        lines += ["", "**Warnings:**"]
        for w in evidence_warnings:
            lines.append(f"- {w}")

    data = {
        "evidence_decision_status": decision_status,
        "evidence_actionable": evidence_actionable,
        "evidence_summary": evidence_summary,
        "evidence_warnings": evidence_warnings,
        "blocking_reasons": blocking_reasons,
        "math_guardrail_events": math_events,
        "citation_verification": citation_result,
        "quantitative_anchors": anchors,
        "evidence_ledger": ledger,
    }

    return {
        "has_content": has_content,
        "markdown": "\n".join(lines),
        "data": data,
    }
