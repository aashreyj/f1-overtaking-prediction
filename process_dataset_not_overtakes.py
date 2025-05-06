import pandas as pd
import random
import numpy as np
import fastf1 as f1
import unicodedata


CHUNK_SIZE = 400
COLUMN_ORDER = [
    "Race",
    "Lap",
    "Position",
    "Overtaker",
    "Overtaken",
    "Turn",
    "Year",
    "Session",
    "OvertakerNumber",
    "OvertakenNumber",
    "NormalizedPosition",
    "X",
    "Y",
    "CornerAngle",
    "AverageCornerSpeed",
    "SpeedDelta",
    "DistanceDelta",
    "OvertakerDrsState",
    "isOvertakerSoft",
    "isOvertakenSoft",
    "isOvertakerMedium",
    "isOvertakenMedium",
    "isOvertakerHard",
    "isOvertakenHard",
    "isOvertakerWet",
    "isOvertakenWet",
    "isOvertakerIntermediate",
    "isOvertakenIntermediate",
    "OvertakerTyreLife",
    "OvertakenTyreLife",
    "IsOvertakerFreshTyre",
    "IsOvertakenFreshTyre",
]

INPUT_FILE = "data/processed_data_overtakes.csv"
OUTPUT_FILE = "data/processed_data_no_overtakes.csv"

# set of session objects
f1.Cache.enable_cache(
    "/data/FastF1_Cache"
)

session_set = dict()


# Function to normalize driver names
def normalize_name(name):
    """Normalize driver names by removing special characters."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )

# Function to get telemetry of closest point
def get_telemetry_of_closest_point(lap, turn_loc):
    """Get telemetry data for the closest point to a given turn location."""
    telemetry = lap.get_telemetry()
    telemetry["DistanceToTurn"] = np.float32(
        (telemetry["X"] - turn_loc[0]) ** 2 + (telemetry["Y"] - turn_loc[1]) ** 2
    )
    return telemetry.loc[telemetry["DistanceToTurn"].idxmin()]

# Function to get tyre age
def get_tyre_age(session, driver_number, compound):
    """Get the total number of laps a tyre has been used in a stint."""
    laps = session.laps.pick_drivers(int(driver_number))
    stint_data = laps.groupby("Stint").agg({"Compound": "first", "LapNumber": "count"})
    stint_data.rename(columns={"LapNumber": "LapsInStint"}, inplace=True)
    return stint_data.loc[stint_data["Compound"] == compound, "LapsInStint"].sum()

# Function to process each row
def process_row(row):
    try:
        # Load the session
        msg = f"Loading session {row['Session']} from {row['Race']}, {int(row['Year'])}"
        print(msg)
        session = f1.get_session(row["Year"], row["Race"], row["Session"])
        if msg in session_set:
            session = session_set[msg]
        else:
            session.load()
            session_set[msg] = session
        
        # Replace overtaker and overtaken names with driver numbers
        driver_map = {
            normalize_name(session.get_driver(drv)["LastName"]).lower(): str(drv)
            for drv in session.drivers
        }
        row["OvertakerNumber"] = driver_map.get(row["Overtaker"].lower(), None)
        row["OvertakenNumber"] = driver_map.get(row["Overtaken"].lower(), None)

        # Pick a random turn number, different from the current one (if provided)
        available_turns = set(range(1, len(session.get_circuit_info().corners) + 1))
        available_turns.discard(row["Turn"])

        row["Turn"] = random.choice(list(available_turns))

        # Normalize position
        row["NormalizedPosition"] = row["Position"] / len(session.drivers)

        # Get turn coordinates and angle
        corners = session.get_circuit_info().corners
        turn_data = corners.query(f"Number == {row['Turn']}")
        row["X"], row["Y"] = turn_data[["X", "Y"]].values[0]
        row["CornerAngle"] = abs(turn_data["Angle"].iloc[0]) / 360

        # Get average speed for the turn
        speeds = [
            get_telemetry_of_closest_point(
                session.laps.pick_fastest(), (row["X"], row["Y"])
            )["Speed"]
            for _ in session.drivers
        ]
        row["AverageCornerSpeed"] = np.mean(speeds)

        # Get telemetry data for overtaker and overtaken
        overtaker_lap = session.laps.pick_drivers(
            int(row["OvertakerNumber"])
        ).pick_laps(int(row["Lap"]))
        overtaken_lap = session.laps.pick_drivers(
            int(row["OvertakenNumber"])
        ).pick_laps(int(row["Lap"]))

        overtaker_telemetry = get_telemetry_of_closest_point(
            overtaker_lap, (row["X"], row["Y"])
        )
        overtaken_telemetry = get_telemetry_of_closest_point(
            overtaken_lap, (row["X"], row["Y"])
        )

        # Compute speed and distance deltas
        row["SpeedDelta"] = np.float32(
            np.float32(overtaker_telemetry["Speed"])
            - np.float32(overtaken_telemetry["Speed"])
        )
        row["DistanceDelta"] = np.float32(
            overtaken_telemetry.get("DistanceToDriverAhead", np.nan)
        )

        # DRS state of overtaker
        row["OvertakerDrsState"] = int(overtaker_telemetry["DRS"] == 2)

        # One-hot encode tyre compounds
        tyre_types = ["Soft", "Medium", "Hard", "Wet", "Intermediate"]
        overtaker_compound = (
            overtaker_lap["Compound"].iloc[0].title()
            if not overtaker_lap.empty
            else "Soft"
        )
        overtaken_compound = (
            overtaken_lap["Compound"].iloc[0].title()
            if not overtaken_lap.empty
            else "Soft"
        )

        for compound in tyre_types:
            row[f"isOvertaker{compound}"] = int(
                compound.lower() in overtaker_compound.lower()
            )
            row[f"isOvertaken{compound}"] = int(
                compound.lower() in overtaken_compound.lower()
            )

        # Compute tyre life fraction
        overtaker_tyre_age = get_tyre_age(
            session, row["OvertakerNumber"], overtaker_lap["Compound"].iloc[0]
        )
        overtaken_tyre_age = get_tyre_age(
            session, row["OvertakenNumber"], overtaken_lap["Compound"].iloc[0]
        )

        row["OvertakerTyreLife"] = np.float32(
            1
            - (
                overtaker_lap["TyreLife"].iloc[0] / overtaker_tyre_age
                if overtaker_tyre_age
                else np.nan
            )
        )
        row["OvertakenTyreLife"] = np.float32(
            1
            - (
                overtaken_lap["TyreLife"].iloc[0] / overtaken_tyre_age
                if overtaken_tyre_age
                else np.nan
            )
        )

        # Check if the tyre was fresh
        row["IsOvertakerFreshTyre"] = int(overtaker_lap.get("FreshTyre", 0).iloc[0])
        row["IsOvertakenFreshTyre"] = int(overtaken_lap.get("FreshTyre", 0).iloc[0])

        # Adjust label for negative sample (overtaking did not happen)
        row["SpeedDelta"] = -abs(row["SpeedDelta"])  # Negative sample: overtaker is slower

        return row
    except Exception as err:
        print(f"Error processing row: {repr(err)}")
        return row        

# Read CSV file data
with pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE) as reader:
    for i, chunk in enumerate(reader):
        # Create a function to generate two new rows for each row
        def generate_rows(row):
            rows = []

            # Create the first new row with a different turn
            new_row_1 = row.copy()
            rows.append(process_row(new_row_1))

            # Create the second new row with another different turn
            new_row_2 = row.copy()
            rows.append(process_row(new_row_2))
            return rows

        # Apply the function to the chunk and flatten the resulting list of lists
        processed_rows = chunk.apply(generate_rows, axis=1).explode().tolist()

        # Convert the processed rows into a DataFrame
        processed_chunk = pd.DataFrame(processed_rows)

        # Save the processed chunk to the output CSV
        processed_chunk[COLUMN_ORDER].to_csv(
            OUTPUT_FILE,
            mode="a",
            header=(i == 0),
            index=False,
        )
        session_set.clear()
