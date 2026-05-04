from __future__ import annotations

import json
import logging
import os
import threading
from typing import Callable

from flask import Flask, jsonify, render_template, request

from .broker import save_portfolio_json
from .config import Aggressiveness, Config

logger = logging.getLogger(__name__)


def create_app(cfg: Config, run_now_fn: Callable) -> Flask:
    app = Flask(__name__, template_folder="templates")
    _running = threading.Event()

    def _read_state() -> dict:
        try:
            with open(cfg.state_file, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "last_intraday_run": None,
                "last_eod_run": None,
                "portfolio_source": "unknown",
                "portfolio_positions": [],
                "cash": 0,
                "actions": [],
                "aggressiveness": cfg.aggressiveness.value,
                "running": False,
            }

    def _patch_state(patch: dict) -> None:
        state = _read_state()
        state.update(patch)
        os.makedirs(os.path.dirname(cfg.state_file), exist_ok=True)
        with open(cfg.state_file, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, default=str)

    @app.route("/")
    def index():
        return render_template("index.html", state=_read_state())

    @app.route("/api/state")
    def api_state():
        return jsonify(_read_state())

    @app.route("/api/aggressiveness", methods=["POST"])
    def set_aggressiveness():
        value = (request.json or {}).get("aggressiveness", "conservative")
        try:
            cfg.aggressiveness = Aggressiveness(value)
        except ValueError:
            return jsonify({"ok": False, "reason": f"Unknown value: {value}"}), 400
        _patch_state({"aggressiveness": value})
        return jsonify({"ok": True})

    @app.route("/api/run", methods=["POST"])
    def trigger_run():
        if _running.is_set():
            return jsonify({"ok": False, "reason": "run already in progress"}), 409

        def _run() -> None:
            _running.set()
            _patch_state({"running": True})
            try:
                run_now_fn()
            except Exception as exc:
                logger.error("On-demand run failed: %s", exc)
            finally:
                _running.clear()
                _patch_state({"running": False})

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/portfolio", methods=["POST"])
    def save_portfolio():
        data = request.json or {}
        try:
            save_portfolio_json(
                cfg.data_dir,
                data.get("positions", []),
                float(data.get("cash", 0)),
            )
        except Exception as exc:
            return jsonify({"ok": False, "reason": str(exc)}), 400
        return jsonify({"ok": True})

    @app.route("/api/broker/check", methods=["POST"])
    def check_broker():
        """Live brokerage connection check. Returns current source, cash, and position count."""
        try:
            portfolio = load_portfolio(cfg.data_dir)
            return jsonify({
                "ok": True,
                "source": portfolio.source,
                "cash": portfolio.cash,
                "positions": len(portfolio.positions),
            })
        except Exception as exc:
            logger.error("Broker check failed: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 500

    return app
