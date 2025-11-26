import json

from tools import get_user_location, get_weather


def pretty_print(label: str, payload: str) -> None:
    print(f"== {label} ==")
    try:
        parsed = json.loads(payload)
        print(json.dumps(parsed, indent=2))
    except Exception:
        print(payload)
    print()


def main() -> None:
    # Call get_user_location and display its envelope
    location_response = get_user_location()
    pretty_print("get_user_location", location_response)

    # Choose a location for weather lookup
    location_query = "New York"
    try:
        data = json.loads(location_response)
        if data.get("success") and isinstance(data.get("result"), dict):
            city = data["result"].get("city") or ""
            country = data["result"].get("country") or ""
            lat = data["result"].get("latitude")
            lng = data["result"].get("longitude")
            # Prefer coordinates if present, otherwise fall back to city,country
            if lat is not None and lng is not None:
                location_query = f"{lat},{lng}"
            elif city or country:
                location_query = f"{city} {country}".strip()
    except Exception:
        pass

    weather_response = get_weather(location_query, days=3)
    pretty_print(f"get_weather ({location_query})", weather_response)


if __name__ == "__main__":
    main()
