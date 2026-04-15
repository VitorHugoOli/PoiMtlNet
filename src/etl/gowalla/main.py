import pandas as pd

from config import CHECKINS_ETL_STEP_1, CHECKINS_ETL_STEP_2
import stage_1
import stage_2
from utils import LOGGER

if __name__ == "__main__":

    checkins = None
    if CHECKINS_ETL_STEP_1.exists():
        LOGGER.info("Loading checkins from ETL step 1.")
        checkins = pd.read_parquet(CHECKINS_ETL_STEP_1)
    else:
        LOGGER.info("Run Stage 1 - Labeling categories.")
        checkins = stage_1.label_categories()

    LOGGER.info("Stage 1 completed.")

    if CHECKINS_ETL_STEP_2.exists():
        LOGGER.info("Loading checkins from ETL step 2.")
        checkins = pd.read_parquet(CHECKINS_ETL_STEP_2)
    else:
        LOGGER.info("Run Stage 2 - Localizing checkins.")
        checkins = stage_2.localize_checkins()

    LOGGER.info("Stage 2 completed.")
