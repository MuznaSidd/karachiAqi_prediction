import numpy as np

# ================= BREAKPOINT TABLES =================

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

NO2_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

O3_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300),
]

SO2_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]


# ================= UTILS =================

def truncate(value, decimals=0):
    factor = 10 ** decimals
    return np.floor(value * factor) / factor


def compute_sub_aqi(conc, breakpoints):
    if conc is None or np.isnan(conc):
        return None

    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= conc <= c_hi:
            return round(
                ((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo
            )

    return 500


# ================= CLASS WRAPPER =================

class EPAAQICalculator:
    def calculate_aqi(self, pm25=None, pm10=None, co=None, no2=None, o3=None, so2=None):
        sub_indexes = {}

        if pm25 is not None:
            pm25 = truncate(pm25, 1)
            sub_indexes["PM2.5"] = compute_sub_aqi(pm25, PM25_BREAKPOINTS)

        if pm10 is not None:
            pm10 = truncate(pm10, 0)
            sub_indexes["PM10"] = compute_sub_aqi(pm10, PM10_BREAKPOINTS)

        if co is not None:
            co = truncate(co, 1)
            sub_indexes["CO"] = compute_sub_aqi(co, CO_BREAKPOINTS)

        if no2 is not None:
            no2 = truncate(no2, 0)
            sub_indexes["NO2"] = compute_sub_aqi(no2, NO2_BREAKPOINTS)

        if o3 is not None:
            o3 = truncate(o3, 0)
            sub_indexes["O3"] = compute_sub_aqi(o3, O3_BREAKPOINTS)

        if so2 is not None:
            so2 = truncate(so2, 0)
            sub_indexes["SO2"] = compute_sub_aqi(so2, SO2_BREAKPOINTS)

        sub_indexes = {k: v for k, v in sub_indexes.items() if v is not None}

        if not sub_indexes:
            return np.nan, None

        dominant = max(sub_indexes, key=sub_indexes.get)
        return sub_indexes[dominant], dominant
