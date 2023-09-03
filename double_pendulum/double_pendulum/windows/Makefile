COMPILATION_FLAG = -I"C:\Windows\SysWOW64\freeglut\include"
LINKAGE_FLAG = -L"C:\Windows\SysWOW64\freeglut\lib" -lfreeglut -lopengl32 -Wl,--subsystem,windows
all:
	g++ -c double_pendulum.cpp $(COMPILATION_FLAG)
	g++ -o double_pendulum.exe double_pendulum.o $(LINKAGE_FLAG) 
clean:
	-del double_pendulum.exe double_pendulum.o