"""Custom exceptions for data checks"""

starters = {"boring_nans" : "Missing CORE_RECOVERY values in boring_intervals:",
            "boring_unfits" : "CORE_RECOVERY is bigger than CORE_INTERVAL in boring_intervals:",
            "boring_overlaps" : "Overlaping intervals found in boring_intervals:",
            "boring_nonincreasing" : "DEPTH_FROM is bigger than the subsequent DEPTH_FROM in boring_intervals:",
            "boring_disordered" : "DEPTH_FROM is bigger than DEPTH_TO in boring_intervals:",
            "lithology_overlaps" : "Overlaping intervals found in core_lithology:",
            "lithology_nonincreasing" : "DEPTH_FROM is bigger than the subsequent DEPTH_FROM in core_lithology:",
            "lithology_disordered" : "DEPTH_FROM is bigger than DEPTH_TO in core_lithology:",
            "lithology_exclusions" : "Following core_lithology intervals are not included in any of boring_intervals:",
            "lithology_unfits" : "Calculated CORE_TOTAL of core_lithology is greater than the corresponding CORE_RECOVERY in boring_intervals:"}

class DataRegularityError(Exception):
    def __init__(self, id, bad_data=''):
        starter = starters.get(id, id)
        message = f"{starter}\n\n{bad_data}"
        super().__init__(message)
