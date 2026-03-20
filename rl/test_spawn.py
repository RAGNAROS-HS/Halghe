import requests
res = requests.post('http://localhost:3000/rl/reset_batch', json={'num_agents': 5})
print([d['state']['player'] for d in res.json()])
