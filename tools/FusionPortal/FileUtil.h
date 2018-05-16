#pragma once
#include <stdio.h>
#include <stdlib.h>

static void LoadFloatData(char* path, float*floatData,int length) {

	FILE *f = fopen(path, "rb");
	if (f == NULL) {
		printf("file not found\n");
	}
	fseek(f, 0, SEEK_END);
	int lSize = ftell(f);
	rewind(f);

	fread(floatData, sizeof(float), 1024 * 1024, f);
	fclose(f);
}