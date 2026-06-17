import os

BASE = r"C:\Users\Administrator\.qclaw\workspace\aijvs-blog\source\_posts"

articles = [
    ("Python编程基础（一）：从零搭建你的AI开发环境.md",
     None,
     "/2026/06/17/Python编程基础（二）：变量、数据类型和基本运算/"),
    ("Python编程基础（二）：变量、数据类型和基本运算.md",
     "/2026/06/17/Python编程基础（一）：从零搭建你的AI开发环境/",
     "/2026/06/17/Python编程基础（三）：NumPy——AI工程师的第一件武器/"),
    ("Python编程基础（三）：NumPy——AI工程师的第一件武器.md",
     "/2026/06/17/Python编程基础（二）：变量、数据类型和基本运算/",
     "/2026/06/17/AI数学基础（一）：线性代数——让机器学会看向量/"),
    ("AI数学基础（一）：线性代数——让机器学会看向量.md",
     "/2026/06/17/Python编程基础（三）：NumPy——AI工程师的第一件武器/",
     "/2026/06/17/AI数学基础（二）：概率论——教机器做不确定的决策/"),
    ("AI数学基础（二）：概率论——教机器做不确定的决策.md",
     "/2026/06/17/AI数学基础（一）：线性代数——让机器学会看向量/",
     "/2026/06/17/AI数学基础（三）：微积分——理解变化的语言/"),
    ("AI数学基础（三）：微积分——理解变化的语言.md",
     "/2026/06/17/AI数学基础（二）：概率论——教机器做不确定的决策/",
     "/2026/05/25/深度学习入门指南：从零开始理解神经网络/"),
]

for fname, prev_url, next_url in articles:
    fp = os.path.join(BASE, fname)
    with open(fp, 'r', encoding='utf-8') as f:
        c = f.read()
    
    # Remove old footer if re-running
    marker = '\u5b66\u4e60\u8def\u5f84\u63a8\u8350\uff1a'
    if marker in c:
        idx = c.find('\n\n---\n\n' + marker)
        if idx > 0:
            c = c[:idx].rstrip()
    
    lines = []
    if prev_url:
        lines.append("[上一篇](" + prev_url + ")")
    if next_url:
        lines.append("[下一篇](" + next_url + ")")
    
    footer = '\n\n---\n\n' + marker + '\n\n' + '\n'.join(lines) + '\n'
    
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(c.rstrip() + footer)
    print('OK')

print('DONE')
