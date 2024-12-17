import os
import subprocess
import time

def run_cmd_old(cmd_string, timeout=5):
    print("command:" + cmd_string)
    try:
        out_bytes = subprocess.check_output("./" + cmd_string, stderr=subprocess.STDOUT, timeout=timeout, shell=True)
        res_code = 0
        msg = out_bytes.decode('utf-8')
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        msg = "[ERROR]CallError ：" + out_bytes.decode('utf-8')
        res_code = e.returncode
    except subprocess.TimeoutExpired as e:
        res_code = 100
        msg = "[ERROR]Timeout:" + str(e)
    except Exception as e:
        res_code = 200
        msg = "[ERROR]Unknown Error:" + str(e)

    return res_code, msg

if __name__ == "__main__":
    # res, msg = run_cmd_old("a.exe")
    # print(res, msg)

    # for filename in os.listdir("./"):
    #     if filename.endswith('.exe'):
    #         compilePopen = subprocess.Popen(filename, shell=True, stderr=subprocess.PIPE)
    #         ret = compilePopen.returncode
    #         if ret != 0:
    #             continue
    #         print(filename)
    #         time.sleep(10)
    #         compilePopen.kill()

    list = []
    for filename in os.listdir("./"):
        if filename.endswith('.out'):
            res, msg = run_cmd_old(filename)
            if res == 0 or res == 100:
                print("rm: " + filename)
                list.append(filename)

    for i in list:
        print(i)
        if os.path.isfile(i):
            os.remove(i)
