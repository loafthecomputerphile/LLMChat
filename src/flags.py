import enum

class ExtractionErrors(enum.IntEnum):
    SUCCESS: int = 0
    FILE_NOT_FOUND: int = 1
    FILE_SIZE_LIMIT: int = 2
    UNKNOWN_ERROR: int = 3
    FILE_TYPE_NOT_RECOGNIZED: int = 4
    
EXTRACTION_ERROR_FLAG: ExtractionErrors = ExtractionErrors.SUCCESS