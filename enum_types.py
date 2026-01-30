from enum import StrEnum


class PlantCsvType(StrEnum):
    MAP = "map"
    DATA = "data"


class SignalType(StrEnum):
    HIGH = "High"
    MIDDLE = "Middle"
    LOW = "Low"
