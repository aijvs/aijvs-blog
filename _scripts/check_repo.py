import requests, json

session = requests.Session()
session.trust_env = False

resp = session.get('https://api.github.com/repos/aijvs/aijvs-blog', timeout=10)
data = resp.json()
print('Status:', resp.status_code)
print('private:', data.get('private'))
print('has_issues:', data.get('has_issues'))
print('visibility:', data.get('visibility'))
