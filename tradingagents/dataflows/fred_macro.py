import os
import requests
import logging

logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get("FRED_API_KEY")

def get_macro_data(series_id: str, start_date: str = None, end_date: str = None, limit: int = 24) -> str:
    """
    Fetches macroeconomic indicators from the Federal Reserve Economic Data (FRED) API.
    """
    if not FRED_API_KEY:
        return "Error: FRED_API_KEY environment variable is not set. Cannot fetch macroeconomic data."
        
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    
    if start_date:
        url += f"&observation_start={start_date}"
    if end_date:
        url += f"&observation_end={end_date}"
        
    # Sort descending so we get latest first if limiting
    url += f"&sort_order=desc&limit={limit}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            error_message = response.json().get("error_message", response.reason)
            logger.error(f"FRED API Error ({response.status_code}): {error_message}")
            return f"Error fetching '{series_id}' from FRED: {error_message}"
            
        data = response.json()
        observations = data.get("observations", [])
        
        if not observations:
            return f"No data found for FRED series: {series_id}"
            
        lines = [f"### FRED Macroeconomic Data: {series_id.upper()}"]
        lines.append(f"Observation Count: {len(observations)} (Showing latest {limit} entries first)")
        lines.append("Date       | Value")
        lines.append("-" * 20)
        
        for obs in observations:
            date = obs.get("date", "Unknown")
            value = obs.get("value", "N/A")
            lines.append(f"{date} | {value}")
            
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"FRED fetching error: {e}")
        return f"Exception occurred while fetching {series_id}: {str(e)}"
