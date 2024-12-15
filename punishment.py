# If we were to make this OOP this is where we use interfaces for different types of punishments
# but since were mainly using one type of punishment just accept a punishment severity.

from enum import Enum
import requests

class Severity(Enum):
    LOW = 1
    MID = 2
    HIGH = 5

def execute(serverity : Severity):
    print("Punishment severity: ", serverity)
    print("Punishment sent...")
    url = "" # TODO: replace with D1 Mini port

    data = {
        "numSprays": serverity.value
    }

    # Custom headers
    headers = {
        "Content-Type": "application/json"
    }

    res = requests.post(url, json=data, headers=headers)

    print(res.status_code)
    print(res.text)

if __name__ == '__main__':
    execute(Severity.LOW)