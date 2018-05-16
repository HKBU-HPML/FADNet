// #pragma once
#include "PointCloudViewer.h"
#include "DepthCameraGrid.h"
#include <errno.h>
#include <stdio.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>

#include "glm/ext.hpp" 
#include "glm/vec3.hpp" 
#include "glm/vec4.hpp" 
#include "glm/mat4x4.hpp" 
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/constants.hpp"


#define VERTEX_SHADER " \
  #version 130\n \
  in vec4 position; \
  in vec4 color; \
  uniform mat4 Model;\
  uniform mat4 View;\
  uniform mat4 Projection;\
  smooth out vec4 vColor; \
  void main() { \
	gl_Position = Projection * View * Model * position; \
	gl_PointSize  = 1;\
    vColor = color; \
	vColor.a = 1;\
  }"

#define FRAGMENT_SHADER " \
  #version 130\n \
  smooth in vec4 vColor; \
  void main() { \
    gl_FragColor = vColor; \
  }"



#define CAM_MODEL_LENGTH 3
#define CAM_MODEL_WIDTH 2
#define CAM_MODEL_HEIGHT 2

const GLfloat cam3DModel[] = {
	0,0,0,														1,0,0,
	CAM_MODEL_WIDTH,	-CAM_MODEL_HEIGHT,	CAM_MODEL_LENGTH,	1,0,0,
	0,0,0,														1,0,0,
	CAM_MODEL_WIDTH,	CAM_MODEL_HEIGHT,	CAM_MODEL_LENGTH,	1,0,0,
	0,0,0,														1,0,0,
	-CAM_MODEL_WIDTH,	CAM_MODEL_HEIGHT,	CAM_MODEL_LENGTH,	1,0,0,
	0,0,0,														1,0,0,
	-CAM_MODEL_WIDTH,	-CAM_MODEL_HEIGHT,	CAM_MODEL_LENGTH,	1,0,0,
	CAM_MODEL_WIDTH,	-CAM_MODEL_HEIGHT,	CAM_MODEL_LENGTH,	1,0,0,
	CAM_MODEL_WIDTH,	CAM_MODEL_HEIGHT,	CAM_MODEL_LENGTH,	1,0,0,
	-CAM_MODEL_WIDTH,	CAM_MODEL_HEIGHT,	CAM_MODEL_LENGTH,	1,0,0,
	-CAM_MODEL_WIDTH,	-CAM_MODEL_HEIGHT,	CAM_MODEL_LENGTH,	1,0,0,
};
int cam3DModelVecCount = 18;
static GLuint camBuffer;
static GLuint  sValuesBuffer;

static GLuint sProgram;
static GLuint sLocPosition;
static GLuint sLocColor;
static GLuint pointCloudBuffer[CAM_NUM];

glm::mat4 ProjectionMtx;
GLint modelP, viewP, projectionP;
float rotationAngle = 0.1f;
int vertNum = 0;
DepthCamera * cams;

DepthCameraGrid *cameraGrid;
int frameId = 0;

GLuint CreateShader(GLenum shaderType, const char* shaderSource){
	GLuint shader = glCreateShader(shaderType);
	glShaderSource(shader, 1, (const GLchar **)&shaderSource, NULL);
	glCompileShader(shader);
	return shader;
}

GLuint CreateProgram(GLuint vertexShader, GLuint fragmentShader){
	GLuint program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);
	return program;
}

void Init(void){
	GLuint vertexShader = CreateShader(GL_VERTEX_SHADER, VERTEX_SHADER);
	GLuint fragmentShader = CreateShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
	sProgram = CreateProgram(vertexShader, fragmentShader);
	for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
		fprintf(stderr, "initial error %d: %s\n", err, gluErrorString(err));
	}
	sLocPosition = glGetAttribLocation(sProgram, "position");
	sLocColor = glGetAttribLocation(sProgram, "color");
	ProjectionMtx = glm::perspectiveFov(45.0f, 1024.0f, 768.0f, 0.001f, 1000.0f);
	for (int i = 0; i < CAM_NUM; i++) {
		glGenBuffers(1, &pointCloudBuffer[i]);
	}
	glGenBuffers(1, &camBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, camBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*cam3DModelVecCount * 6, cam3DModel, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}



void DrawCameras(glm::mat4 viewMtx) {
	glUseProgram(sProgram);

	modelP = glGetUniformLocation(sProgram, "Model");
	viewP = glGetUniformLocation(sProgram, "View");
	projectionP = glGetUniformLocation(sProgram, "Projection");

	glUniformMatrix4fv(viewP, 1, GL_FALSE, glm::value_ptr(viewMtx));
	glUniformMatrix4fv(projectionP, 1, GL_FALSE, glm::value_ptr(ProjectionMtx));

	glBindBuffer(GL_ARRAY_BUFFER, camBuffer);
	glEnableVertexAttribArray(sLocPosition);
	glVertexAttribPointer(sLocPosition, 3, GL_FLOAT, GL_FALSE, 24, (void *)0);
	glEnableVertexAttribArray(sLocColor);
	glVertexAttribPointer(sLocColor, 3, GL_FLOAT, GL_FALSE, 24, (void *)12);

	for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
		fprintf(stderr, "before camera error %d: %s\n", err, gluErrorString(err));
	}

	for (int i = 0; i < CAM_NUM; i++) {
		//glm::mat4 newModel = glm::make_mat4(cams[i].transform);
		//newModel = glm::transpose(newModel);
		glUniformMatrix4fv(modelP, 1, GL_FALSE, glm::value_ptr(cams[i].transform));
		glDrawArrays(GL_LINE_STRIP, 0, cam3DModelVecCount);
	}
	for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
		fprintf(stderr, "after camera error %d: %s\n", err, gluErrorString(err));
	}
	glDisableVertexAttribArray(sLocPosition);
	glDisableVertexAttribArray(sLocColor);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);
}

void DrawPointClouds(glm::mat4 viewMtx) {
	glUseProgram(sProgram);
	glEnable(GL_PROGRAM_POINT_SIZE);

	modelP = glGetUniformLocation(sProgram, "Model");
	viewP = glGetUniformLocation(sProgram, "View");
	projectionP = glGetUniformLocation(sProgram, "Projection");

	glUniformMatrix4fv(viewP, 1, GL_FALSE, glm::value_ptr(viewMtx));
	glUniformMatrix4fv(projectionP, 1, GL_FALSE, glm::value_ptr(ProjectionMtx));

	for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
		fprintf(stderr, "before cloud %d: %s\n", err, gluErrorString(err));
	}

	for (int i = 0; i < CAM_NUM; i++) {
		//glm::mat4 newModel = glm::make_mat4(cams[i].transform);
		//newModel = glm::transpose(newModel);
		glUniformMatrix4fv(modelP, 1, GL_FALSE, glm::value_ptr(cams[i].transform));
		glBindBuffer(GL_ARRAY_BUFFER, pointCloudBuffer[i]);
		//glBindBuffer(GL_ARRAY_BUFFER, sValuesBuffer);
		glEnableVertexAttribArray(sLocPosition);
		glVertexAttribPointer(sLocPosition, 3, GL_FLOAT, GL_FALSE, 24, (void *)0);
		glEnableVertexAttribArray(sLocColor);
		glVertexAttribPointer(sLocColor, 3, GL_FLOAT, GL_FALSE, 24, (void *)12);
		glDrawArrays(GL_POINTS, 0, vertNum);

	}
	for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
		fprintf(stderr, "after cloud %d: %s\n", err, gluErrorString(err));
	}
	glDisableVertexAttribArray(sLocPosition);
	glDisableVertexAttribArray(sLocColor);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);
}

void GLUT_display(void){
	//update point clouds , just for testing
	
	vertNum = IMAGE_WIDTH * IMAGE_HEIGHT / SAMPLE_STEP / SAMPLE_STEP;
	for (int i = 0; i < CAM_NUM; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, pointCloudBuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertNum * 6, cameraGrid->vertexData[frameId][i], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	frameId++;
	if (frameId >= FRAME) {
		frameId = 0;
	}
	//calculate view matrix
	rotationAngle += 0.01;
	float x = sinf(rotationAngle) * 60;
	float z = cosf(rotationAngle) * 60;

	glm::mat4 viewMtx;
	viewMtx = glm::lookAt(glm::vec3(x, 15.0f, z), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	DrawCameras(viewMtx);
	DrawPointClouds(viewMtx);
	glutSwapBuffers();
}

void GLUT_reshape(int w, int h){
	glViewport(0, 0, w, h);
}

void GLUT_Timer(int value){
	glutPostRedisplay();
	glutTimerFunc(33, GLUT_Timer, 1);
}

PointCloudViewer::PointCloudViewer() {
	int my_argc = 0;
	glutInit(&my_argc, NULL);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(1024, 768);
	glutInitWindowPosition(200, 100);
	glutCreateWindow(NULL);
	glewInit();
	glutDisplayFunc(GLUT_display);
	glutReshapeFunc(GLUT_reshape);
	glutTimerFunc(33, GLUT_Timer, 1);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	Init();
	
}

void PointCloudViewer::UpdatePointData(DepthCamera *cameras,float **data,int vertciesNum) {
	/*int s = 10;
	float *testdata = (float*)malloc(sizeof(float) * s * s * 6);
	for (int y = 0; y < s; y++) {
		for (int x = 0; x < s; x++) {
			int id = (y * s + x) * 6;
			testdata[id] = (x-.5*s)*.1f ;
			testdata[id + 1] = (y-.5*s)*.1f;
			testdata[id + 2] = 0;
			testdata[id + 3] = 0;// y*1.0 / s;
			testdata[id + 4] = 0;// x*1.0 / s;
			testdata[id + 5] = 0;
		}
	}
	vertNum = s * s*2-2;
	glGenBuffers(1, &sValuesBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, sValuesBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * s * s * 6, testdata, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return;*/
	cams = cameras;
	vertNum = vertciesNum*2;
	for (int i = 0; i < CAM_NUM; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, pointCloudBuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertciesNum * 6, data[i], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void PointCloudViewer::RegistGrid(void * grid) {
	cameraGrid = (DepthCameraGrid *)grid;
	cams = cameraGrid->cams;
}

void PointCloudViewer::LoopGlut() {
	glutMainLoop();
}
