import os

def getfilename():
    file = open("validated_bugs")
    trace = []
    while 1:
        lines = file.readlines(100000)
        if not lines:
            break
        for line in lines:
                trace.append(line[:-1])
    file.close()
    return trace


if __name__ == "__main__":
    path = "./coreutils-8.24-lava-safe/lava-install/bin/base64 -d"
    files = getfilename()
    count = 0
    for i in range(884):
        q = str(i + 1)
        if q not in files:
            write_file = "/home/ict2080ti/liuli_repo/logs/exp/base64/pos/" + q + ".log"
            target_file = "./inputs/utmp-fuzzed-" + q + ".b64"
            cmd = "qemu-x86_64 -d nochain,exec,out_asm -D " + write_file + " -- " + path + " " + target_file
            count += 1
            #print(cmd)
            os.system(cmd)
            if count > 40:
                break

        # else:
        #     write_file = "/home/ict2080ti/liuli_repo/logs/exp/base64/pos/" + str(q) + ".log"


    # for filename in os.listdir("./"):
    #     if filename.endswith('.out'):
    #         os.system("qemu-x86_64 -d nochain,exec,out_asm -D /home/ict2080ti/liuli_repo/logs/neg/" + filename + ".log -- ./" + filename)