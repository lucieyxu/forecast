from datetime import datetime

def calculate_days_between_dates(start_date_str: str, end_date_str: str) -> int:
    """
    Calculates the number of days between two dates given as strings.

    Args:
        start_date_str: The starting date in "YYYY-MM-DD" format.
        end_date_str: The ending date in "YYYY-MM-DD" format.

    Returns:
        The number of days between the two dates.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return (end_date - start_date).days


def calculate_weeks_between_dates(start_date_str: str, end_date_str: str) -> int:
    """
    Calculates the number of weeks between two dates given as strings.

    Args:
        start_date_str: The starting date in "YYYY-MM-DD" format.
        end_date_str: The ending date in "YYYY-MM-DD" format.

    Returns:
        The number of weeks between the two dates.
    """
    weeks = calculate_days_between_dates(start_date_str, end_date_str) // 7
    return weeks
