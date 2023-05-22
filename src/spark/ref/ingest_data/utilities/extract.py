import os 
import datetime

def get_today() :
    date_format = '%Y%m%d'
    today = datetime.date.today()
    formatted_date = today.strftime(date_format)
    return str(formatted_date)
