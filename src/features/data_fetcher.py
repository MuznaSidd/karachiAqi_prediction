import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from datetime import datetime, timedelta
import yaml


class AQIDataFetcher:
    """
    Fetches 1 year of:
    - Air Quality data (pollutants)
    - Weather data (using Historical Weather API)
    And merges them on hourly timestamps
    """

    def __init__(self):
        # Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.lat = config["location"]["latitude"]
        self.lon = config["location"]["longitude"]
        self.city = config["location"]["city"]

        # Open-Meteo client (cache + retry)
        cache = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

        self.air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        
        self.weather_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    # --------------------------------------------------
    # AIR QUALITY (HISTORICAL)
    # --------------------------------------------------
    def _fetch_air_quality(self, days: int) -> pd.DataFrame:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        print(f"📥 Fetching AIR QUALITY from {start_date} to {end_date}")

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "ammonia",
            ],
        }

        response = self.client.weather_api(self.air_quality_url, params=params)[0]
        hourly = response.Hourly()

        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        data = {
            "timestamp": timestamps,
            "pm2_5": hourly.Variables(0).ValuesAsNumpy(),
            "pm10": hourly.Variables(1).ValuesAsNumpy(),
            "carbon_monoxide": hourly.Variables(2).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(3).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(4).ValuesAsNumpy(),
            "ozone": hourly.Variables(5).ValuesAsNumpy(),
            "ammonia": hourly.Variables(6).ValuesAsNumpy(),
        }

        df = pd.DataFrame(data)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        print(f"✅ Air-quality rows: {len(df)}")
        return df

    # --------------------------------------------------
    # WEATHER (HISTORICAL FORECAST API)
    # --------------------------------------------------
    def _fetch_weather(self, days: int) -> pd.DataFrame:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        print(f"🌦️ Fetching WEATHER from {start_date} to {end_date}")

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "surface_pressure",
            ],
        }

        response = self.client.weather_api(self.weather_url, params=params)[0]
        hourly = response.Hourly()

        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        data = {
            "timestamp": timestamps,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(3).ValuesAsNumpy(),
            "surface_pressure": hourly.Variables(4).ValuesAsNumpy(),
        }

        df = pd.DataFrame(data)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        print(f"✅ Weather rows: {len(df)}")
        return df

    # --------------------------------------------------
    # PUBLIC METHOD (MERGED DATA)
    # --------------------------------------------------
    def fetch_historical_data(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical data for specified days.
        
        Args:
            days: Number of days (default 365)
        
        Returns:
            Merged DataFrame with air quality + weather
        """
        aq_df = self._fetch_air_quality(days)
        weather_df = self._fetch_weather(days)

        print("🔗 Merging air-quality + weather data...")
        df = aq_df.merge(weather_df, on="timestamp", how="inner")

        print(f"🎯 Final merged rows: {len(df)}")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df


# --------------------------------------------------
# TEST RUN
# --------------------------------------------------
if __name__ == "__main__":
    fetcher = AQIDataFetcher()
    df = fetcher.fetch_historical_data(days=365)
    print(df.head())
    print(df.tail())