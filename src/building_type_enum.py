from enum import Enum


class BuildingType(Enum):
    SINGLE_FAMILY = "single_family"
    APARTMENT_CONDO = "apartment_condo"
    COMMERCIAL = "commercial"
    MIXED_USE = "mixed_use"
    EMPTY_LOT = "empty_lot"
    UNKNOWN = "unknown"
