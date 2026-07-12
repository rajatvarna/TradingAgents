"""SEC EDGAR full-text search vendor: query building, output formatting, and
error handling. All API access is mocked, so these run without a network
connection.
"""
import unittest
from unittest import mock

import pytest

from tradingagents.dataflows import sec_edgar

_SEARCH_RESPONSE = {
    "hits": {
        "hits": [
            {
                "_id": "0000320193-24-000123:aapl-20240928.htm",
                "_source": {
                    "display_names": ["Apple Inc. (AAPL)"],
                    "form": "10-K",
                    "file_date": "2024-11-01",
                    "ciks": ["0000320193"],
                },
            }
        ]
    }
}


@pytest.mark.unit
class SecEdgarSearchTests(unittest.TestCase):
    def test_empty_query_returns_error_without_network_call(self):
        with mock.patch.object(sec_edgar, "_request") as m:
            report = sec_edgar.search_filings("   ")
        m.assert_not_called()
        self.assertTrue(report.startswith("ERROR:"))

    def test_search_formats_report_with_hits(self):
        with mock.patch.object(sec_edgar, "_request", return_value=_SEARCH_RESPONSE) as m:
            report = sec_edgar.search_filings("material weakness", ticker="AAPL", forms="10-K")

        params = m.call_args[0][0]
        self.assertEqual(params["q"], "material weakness")
        self.assertEqual(params["ciks"], "AAPL")
        self.assertEqual(params["forms"], "10-K")
        self.assertIn("Apple Inc.", report)
        self.assertIn("10-K", report)
        self.assertIn("2024-11-01", report)
        self.assertIn("sec.gov/Archives/edgar/data/320193", report)

    def test_search_no_hits_returns_readable_message(self):
        with mock.patch.object(sec_edgar, "_request", return_value={"hits": {"hits": []}}):
            report = sec_edgar.search_filings("some rare phrase")
        self.assertIn("No matching filings found.", report)

    def test_date_range_params_are_set_when_provided(self):
        with mock.patch.object(sec_edgar, "_request", return_value={"hits": {"hits": []}}) as m:
            sec_edgar.search_filings("going concern", date_from="2024-01-01", date_to="2024-12-31")
        params = m.call_args[0][0]
        self.assertEqual(params["startdt"], "2024-01-01")
        self.assertEqual(params["enddt"], "2024-12-31")

    def test_request_sends_user_agent_header(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"hits": {"hits": []}}
        with mock.patch("requests.get", return_value=response) as m:
            sec_edgar._request({"q": "test"})
        headers = m.call_args.kwargs["headers"]
        self.assertIn("User-Agent", headers)

    def test_request_raises_on_error_status(self):
        response = mock.Mock(status_code=500, text="internal error")
        with mock.patch("requests.get", return_value=response):
            with self.assertRaises(ValueError):
                sec_edgar._request({"q": "test"})
