import re
import subprocess
import json
import progressbar # pip install progressbar2

"""[Initialize] Eigen: 15ms
[Initialize] Mine: 2ms
[Access all elements] Eigen: 79ms
[Access all elements] Mine: 37ms
[Modify elements] Eigen: 6764ms
[Modify elements] Mine: 19ms
"""

pattern = re.compile(r'^\[.*\] [a-zA-Z]*: ([0-9]*)ms[\s]*$', re.M)
err_pattern = re.compile(r'\[!!ERROR\]')

density = {
    'dense': [],
    'access_eign': [],
    'access_mine': [],
    'modify_eign': [],
    'modify_mine': [],
}

#for i in range(1, 80):
for i in progressbar.progressbar(range(1, 80), redirect_stdout=True):
    result = subprocess.run(['./lab3.exe', '-b', '-d', '5', '-s', str(i*20)], stdout=subprocess.PIPE)
    out = result.stdout.decode('utf-8')
    if len(err_pattern.findall(out)):
        print(out)
        print(err_pattern.findall(out))
        print('Error encountered')
        exit(0)

    ms = list(map(lambda x: int(x) , pattern.findall(out)))
    
    density['dense'].append(i)
    density['access_eign'].append(ms[2])
    density['access_mine'].append(ms[3])
    density['modify_eign'].append(ms[4])
    density['modify_mine'].append(ms[5])
    
    

with open('lab3.benchmark.json', 'wt') as f:
    json.dump(density, f)
with open('lab3.benchmark.pretty.json', 'wt') as f:
    s = ''
    for key, val in density.items():
        s += '"{}":{},\n'.format(key, json.dumps(val))
    s = '{\n' + s[:-1] + '\n}'
    f.write(s)

