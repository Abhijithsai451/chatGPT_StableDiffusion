from pyChatGPT import ChatGPT
from pprint import pprint
import py_secrets

session_token = py_secrets.session_token
api = ChatGPT(session_token)
response = api.send_message("Explain Stable DIffusion model to a layman")

pprint(response)
