def calculate_parking_fee(entry_time, exit_time, rate_per_hour=2.0):
    """Calculates parking fee based on duration. First 15 mins are free."""
    if not entry_time or not exit_time:
        return 0.0
    
    duration = exit_time - entry_time
    hours = duration.total_seconds() / 3600
    
    # First 15 mins free
    if hours < 0.25: 
        return 0.0
        
    return round(hours * rate_per_hour, 2)
