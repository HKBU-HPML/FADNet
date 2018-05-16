# C compiler
CC = g++
CC_FLAGS = -O2
INCLUDE_PATH = -I./include/opencv/include -I./include/glm/include -I./include/GLEW/include -I./include/GLFW/include -I./include/freeglut/include
LIB_PATH = -L./lib # -L/usr/local/hdf5/lib/
LINK_LIB = -lopencv_world -lGL -lGLU -lglut -lGLEW -lm -lhdf5


Main: DepthCamera.o DepthCameraGrid.o PointCloudViewer.o 
	$(CC) $(CC_FLAGS) $(INCLUDE_PATH) $(LIB_PATH) DepthCamera.o DepthCameraGrid.o PointCloudViewer.o Main.cpp -o Main $(LINK_LIB)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

DepthCamera.o: DepthCamera.cpp DepthCamera.h
	$(CC) $(CC_FLAGS) $(INCLUDE_PATH) DepthCamera.cpp -c
	
DepthCameraGrid.o: DepthCameraGrid.cpp DepthCameraGrid.h
	$(CC) $(CC_FLAGS) $(INCLUDE_PATH) DepthCameraGrid.cpp -c

PointCloudViewer.o: PointCloudViewer.cpp PointCloudViewer.h
	$(CC) $(CC_FLAGS) $(INCLUDE_PATH) PointCloudViewer.cpp -c

clean:
	rm -f *.o Main
