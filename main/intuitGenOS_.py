import json
import requests
import os
import genstudiopy
import uuid


def get_offline_ticket_header(e2e_offline_job_id=9341451530140128):
    gql_query = "mutation identitySignInInternalApplicationWithPrivateAuth($input: Identity_SignInApplicationWithPrivateAuthInput!) {\n\n    identitySignInInternalApplicationWithPrivateAuth(input: $input) {\n        accessToken {\n            token\n            tokenType\n            expiresInSeconds\n        }\n        refreshToken {\n            token\n            tokenType\n            expiresInSeconds\n        }\n        accountContext {\n            accountId\n          profileId\n            namespace\n            pseudonymId\n        }\n   authorizationHeader\n }\n}"
    app_id = "Intuit.isr.ai.ml.cptestclient"
    e2e_access_url = "https://identityinternal-e2e.api.intuit.com/v1/graphql"
    intuit_tid = str(uuid.uuid4())

    body = {
            "query": gql_query,
            "variables": {
                "input": {
                    "profileId": e2e_offline_job_id

                }
            }
        }

    headers = {
            "intuit_tid": intuit_tid,
            "Authorization": f"Intuit_IAM_Authentication "
                              f"intuit_appid={app_id},"
                              f"intuit_app_secret=preprd83KyHGNNyfhaHqolUdBgy91ObZrVS9GAgg",
            "Content-Type": "application/json"
        }

    response = requests.post(url=e2e_access_url, data=json.dumps(body), headers=headers)
    offlineAuthHeader = response.json()

    assert response.status_code == 200, \
          f"error during offline token generation, status_code={response.status_code}, {offlineAuthHeader}"

    offlineAuthHeader = offlineAuthHeader["data"]["identitySignInInternalApplicationWithPrivateAuth"]["authorizationHeader"]

    return offlineAuthHeader


def extract_auths(auth_header):
    valid_elements = auth_header.split(" ")[1].split(',')
    auth_id = ""
    ticket = ""

    for elem in valid_elements:

        if "intuit_userid" in elem:
            auth_id = elem.split("=")[1]
        if "intuit_token=" in elem:
            ticket = elem.split("=")[1]

    return {"auth_id":auth_id,"ticket":ticket}


app_id = "Intuit.isr.ai.ml.cptestclient"
e2e_app_secret = "preprd83KyHGNNyfhaHqolUdBgy91ObZrVS9GAgg"
identityAuthHeader = get_offline_ticket_header()
authorization = identityAuthHeader + f",intuit_appid={app_id},intuit_app_secret={e2e_app_secret}"
headers = extract_auths(identityAuthHeader)

# os.environ['INTUIT_APP_ID'] = ''
from genstudiopy import config
result = genstudiopy.ChatFilm.create(
      model="chat-film",
      version="3",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "How is the day"}
      ],
        metadata=dict(
        env='e2e',
        app_id=app_id,
        app_secret=e2e_app_secret,
        auth_id=headers['auth_id'],
        ticket=headers['ticket']
      )
  )
print(result)