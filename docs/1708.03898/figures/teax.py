import sys,os,time
for _ in range(10**6):
	print("\033[92m--------->>> BUILD #{}\033[0m".format(_))
	idx, flag = sys.argv[1], False
	if len(sys.argv) >= 3:
		flag = True if sys.argv[2] == "-s" else False
	os.system("teax build {}.tex".format(idx))
	os.system("convert           \
	-verbose       \
	-density 300   \
	-trim          \
	{}.pdf      \
	-quality 100   \
	-flatten       \
	-sharpen 0x1.0 \
	{}.jpg".format(idx, idx))
	os.system("jpegoptim {}.jpg".format(idx))
	os.system("rm -rf .teax/")
	if flag == True:
		os.system("rm {}.pdf".format(idx))
		break
	if _ % 10 == 0: os.system("rm {}.pdf".format(idx))
	time.sleep(2)
