"""Contains specific Exceptions."""


STARTERS = {
    "boring_nans" : "Missing CORE_RECOVERY values in boring_intervals:\n\n{}",
    "boring_unfits" : "CORE_RECOVERY is bigger than CORE_INTERVAL in boring_intervals:\n\n{}",
    "boring_overlaps" : "Overlaping intervals found in boring_intervals:\n\n{}",
    "boring_nonincreasing" : "DEPTH_FROM is bigger than the subsequent DEPTH_FROM in boring_intervals:\n\n{}",
    "boring_disordered" : "DEPTH_FROM is bigger than DEPTH_TO in boring_intervals:\n\n{}",
    "lithology_overlaps" : "Overlaping intervals found in core_lithology:\n\n{}",
    "lithology_nonincreasing" : "DEPTH_FROM is bigger than the subsequent DEPTH_FROM in core_lithology:\n\n{}",
    "lithology_disordered" : "DEPTH_FROM is bigger than DEPTH_TO in core_lithology:\n\n{}",
    "lithology_exclusions" : "Following core_lithology intervals are not included in any of boring_intervals:\n\n{}",
    "lithology_unfits" : "Calculated CORE_TOTAL is greater than corresponding CORE_RECOVERY in boring_intervals:\n\n{}",
    "missing_files" : "Following files from {} are missing in {}:\n\n{}",
    "duplicate_files" : "Duplicate file names in {}",
    "different_extensions" : "File extensions from {} have different extension length"
    }


class SkipWellException(Exception):
    """Raised if a well should be dropped from a batch."""


class DataRegularityError(SkipWellException):
    """Raised if any data regularity checks are not passed."""
    def __init__(self, error_id, *args):
        starter = STARTERS.get(error_id, error_id)
        message = starter.format(*args)
        super().__init__(message)
