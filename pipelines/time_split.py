from datetime import datetime, timedelta

from helpers.date import calculate_weeks_between_dates


class TimeSplit:
    def __init__(
        self,
        train_start_date: str,
        test_start_date: str,
        test_end_date: str,
        horizon: int,
    ):
        self.train_start_date = train_start_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

        test_start_datetime = datetime.strptime(test_start_date, "%Y-%m-%d").date()
        val_start_datetime = test_start_datetime - timedelta(weeks=horizon)

        self.val_start_date = val_start_datetime.strftime("%Y-%m-%d")
        self.train_length = calculate_weeks_between_dates(
            train_start_date,
            self.val_start_date,
        )
        self.test_length = calculate_weeks_between_dates(
            test_start_date,
            test_end_date,
        )
        self.val_length = calculate_weeks_between_dates(
            self.val_start_date,
            self.test_start_date,
        )
