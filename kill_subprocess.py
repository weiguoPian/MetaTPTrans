import subprocess

cmd = 'ps aux|grep __main_'

(status, output) =subprocess.getstatusoutput(cmd)
output = output.split('\n')

for line in output:
    subprocess_num = line.split(' ')[4]
    print(subprocess_num)
    cmd_kill = 'kill -9 {}'.format(subprocess_num)
    subprocess.call(cmd_kill, shell=True)