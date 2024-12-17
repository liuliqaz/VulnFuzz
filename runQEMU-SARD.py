import os

if __name__ == "__main__":
    for filename in os.listdir("./"):
        if filename.endswith('.out'):
            os.system("qemu-x86_64 -d nochain,exec,out_asm -D /home/ict2080ti/liuli_repo/logs/neg/" + filename + ".log -- ./" + filename)