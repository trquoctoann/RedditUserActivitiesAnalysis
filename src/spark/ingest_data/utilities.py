import requests
import configparser
import urllib3
import datetime
import mysql.connector

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
configFilePath = 'project.conf'

# create the authentication header to query 
def get_reddit_bearer_token():
    # get spotify api info
    parser = configparser.ConfigParser()
    parser.read(configFilePath)
    auth_url = parser.get("reddit_api_config", "reddit_auth_url")
    username = parser.get("reddit_api_config", "reddit_api_username")
    password = parser.get("reddit_api_config", "reddit_api_password")
    app_id = parser.get("reddit_api_config", "reddit_api_app_id")
    app_secret = parser.get("reddit_api_config", "reddit_api_app_secret")
    user_agent = parser.get("reddit_api_config", "reddit_api_user_agent")

    # connect to API
    payload = {'username': username, 'password': password, 'grant_type': 'password'}
    auth = requests.auth.HTTPBasicAuth(app_id, app_secret)
    response = requests.post(auth_url, data = payload, headers={'user-agent': user_agent}, auth = auth)
    if response.status_code != 200:
        print("Error Response Code: " + str(response.status_code))
        raise Exception(response.status_code, response.text)
    access_token = response.json()["access_token"]
    return "Bearer " + access_token

# convert created_utc value to datetime
def format_reddit_created_date(date):
    date_format = '%Y-%m-%d %H:%M:%S'
    dt = datetime.datetime.fromtimestamp(date)
    formatted_date = dt.strftime(date_format)
    return formatted_date

# send request to get new post
def send_request_reddit_get_new_post(url, access_token = get_reddit_bearer_token()):
    parser = configparser.ConfigParser()
    parser.read(configFilePath)
    user_agent = parser.get("reddit_api_config", "reddit_api_user_agent")
    parameters = {'limit' : 100}
    response = requests.get(url, headers = {'Authorization' : access_token, 'user-agent': user_agent}, params = parameters)
    if response.status_code != 200:
        print("Error Response Code: " + str(response.status_code))
        raise Exception(response.status_code, response.text)
    return response.json(), response.status_code

def check_database_existence(mycursor) :
    mycursor.execute('show databases')
    exist = False
    for database in mycursor : 
        if database[0] == 'historyDB' : 
            exist = True 
            break
    myresult = mycursor.fetchall()
    return exist

def check_table_existence(mycursor) : 
    mycursor.execute('show tables')
    exist = False
    for table in mycursor : 
        if table[0] == 'executionHistory' : 
            exist = True 
            break
    myresult = mycursor.fetchall()
    return exist

def mysql_connector() : 
    connector = mysql.connector.connect(host = 'web-database', user = 'root', password = '123', port = int(3306))
    mycursor = connector.cursor()
    database_existence = check_database_existence(mycursor)
    if database_existence is False : 
        mycursor.execute('CREATE DATABASE historyDB')
    mycursor.execute('USE historyDB')
    table_existence = check_table_existence(mycursor)
    if table_existence is False : 
        mycursor.execute('CREATE TABLE executionHistory (rBusiness_last_time DATETIME, rTechnology_last_time DATETIME, \
                         rSports_last_time DATETIME, rWorldNews_last_time DATETIME)')
        dml_insertion = "INSERT INTO executionHistory (rBusiness_last_time, rTechnology_last_time, rSports_last_time, rWorldNews_last_time) \
                         VALUES (%s, %s, %s, %s)"
        value = [('2023-05-10 00:00:00', '2023-05-10 00:00:00', '2023-05-10 00:00:00', '2023-05-10 00:00:00')]
        mycursor.executemany(dml_insertion, value)
        connector.commit()
    return connector

def get_last_execution_date() : 
    connector = mysql_connector()
    mycursor = connector.cursor()
    mycursor.execute('USE historyDB')
    get_last_execution = "SELECT rBusiness_last_time, rTechnology_last_time, rSports_last_time, rWorldNews_last_time \
                          FROM executionHistory ORDER BY rBusiness_last_time DESC LIMIT 1"
    mycursor.execute(get_last_execution)
    result = mycursor.fetchone()
    return result

def save_new_execution_date(rBusiness_execution_time, rTechnology_execution_time, 
                            rSports_execution_time, rWorldNews_execution_time) : 
    connector = mysql_connector()
    mycursor = connector.cursor()
    mycursor.execute('USE historyDB')
    dml_insertion = "INSERT INTO executionHistory (rBusiness_last_time, rTechnology_last_time, rSports_last_time, rWorldNews_last_time) \
                     VALUES (%s, %s, %s, %s)"
    value = [(rBusiness_execution_time, rTechnology_execution_time, rSports_execution_time, rWorldNews_execution_time)]
    mycursor.executemany(dml_insertion, value)
    connector.commit()
    print("Extraction datetime added to History-database")