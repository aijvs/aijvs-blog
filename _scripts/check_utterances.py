import re

post = r'public/2026/06/17/Python编程基础（一）：从零搭建你的AI开发环境/index.html'
with open(post, 'r', encoding='utf-8') as f:
    content = f.read()

# Find utterances configuration block
match = re.search(r'const\s+loadUtterances.*?window\.uttr\s*=\s*uttr', content, re.DOTALL)
if match:
    block = match.group()
    # Extract key params
    repo_match = re.search(r"repo:\s*'([^']+)'", block)
    issue_match = re.search(r"issue_term:\s*'([^']+)'", block)
    theme_match = re.search(r"theme:\s*'([^']+)'", block)
    
    print('Utterances config found:')
    if repo_match:
        print(f'  repo: {repo_match.group(1)}')
    if issue_match:
        print(f'  issue_term: {issue_match.group(1)}')
    if theme_match:
        print(f'  theme: {theme_match.group(1)}')
else:
    print('LoadUtterances function not found')
    # Try alternative pattern
    match = re.search(r'utteranc\.es/client\.js.*?}', content, re.DOTALL)
    if match:
        print('Alternative pattern found:')
        print(match.group()[:300])
