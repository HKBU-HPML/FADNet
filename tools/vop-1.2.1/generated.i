void inSQRTv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_sqrt_ps(_mm_load_ps((float*)(a0))));
    result += 4;
    a0++;
  }
}
void inNEGv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_xor_ps(_mm_load_ps((float*)(a0)),(__m128)_mm_set_epi32(0x80000000,0x80000000,0x80000000,0x80000000)));
    result += 4;
    a0++;
  }
}
void inNOTv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_xor_ps(_mm_load_ps((float*)(a0)),(__m128)_mm_set_epi32(0xffffffff,0xffffffff,0xffffffff,0xffffffff)));
    result += 4;
    a0++;
  }
}
void inABSv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_and_ps(_mm_load_ps((float*)(a0)),(__m128)_mm_set_epi32(0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff)));
    result += 4;
    a0++;
  }
}
void inINTv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, (__m128)_mm_cvtps_epi32(_mm_load_ps((float*)(a0))));
    result += 4;
    a0++;
  }
}
void inFLOORv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, (__m128)_mm_cvtepi32_ps(_mm_cvtps_epi32(_mm_load_ps((float*)(a0)))));
    result += 4;
    a0++;
  }
}
void inCHARv(int *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  for (i = ((l+3)>>2); i; i--) {
    *result++ = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi16(_mm_cvtps_epi32(_mm_load_ps((float*)(a0))),(__m128i)(__m128)_mm_set_epi32(0,0,0,0)),(__m128i)(__m128)_mm_set_epi32(0,0,0,0)));
    a0++;
  }
}
void inMINvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_min_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inMINvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_min_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inMINsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_min_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inMAXvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_max_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inMAXvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_max_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inMAXsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_max_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inADDvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inADDvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inADDsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inSUBvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_sub_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inSUBvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_sub_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inSUBsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_sub_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inMULvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_mul_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inMULvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_mul_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inMULsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_mul_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inDIVvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_div_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inDIVvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_div_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inDIVsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_div_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inANDvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_and_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inANDvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_and_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inANDsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_and_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inORvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inORvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inORsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inXORvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_xor_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inXORvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_xor_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inXORsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_xor_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inEQvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpeq_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inEQvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpeq_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inEQsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpeq_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inNEvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpneq_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inNEvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpneq_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inNEsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpneq_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inLTvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmplt_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inLTvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmplt_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inLTsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmplt_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inGEvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpnlt_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inGEvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpnlt_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inGEsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpnlt_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inLEvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmple_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inLEvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmple_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inLEsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmple_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inGTvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpnle_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))));
    result += 4;
    a0++,a1++;
  }
}
void inGTvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpnle_ps(_mm_load_ps((float*)(a0)),a1));
    result += 4;
    a0++;
  }
}
void inGTsv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_cmpnle_ps(a0,_mm_load_ps((float*)(a1))));
    result += 4;
    a1++;
  }
}
void inWHEREvvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_andnot_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a2))),_mm_and_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1)))));
    result += 4;
    a0++,a1++,a2++;
  }
}
void inWHEREvvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_andnot_ps(_mm_load_ps((float*)(a0)),a2),_mm_and_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1)))));
    result += 4;
    a0++,a1++;
  }
}
void inWHEREvsv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_andnot_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a2))),_mm_and_ps(_mm_load_ps((float*)(a0)),a1)));
    result += 4;
    a0++,a2++;
  }
}
void inWHEREsvv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_andnot_ps(a0,_mm_load_ps((float*)(a2))),_mm_and_ps(a0,_mm_load_ps((float*)(a1)))));
    result += 4;
    a1++,a2++;
  }
}
void inWHEREsvs(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_andnot_ps(a0,a2),_mm_and_ps(a0,_mm_load_ps((float*)(a1)))));
    result += 4;
    a1++;
  }
}
void inWHEREssv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_andnot_ps(a0,_mm_load_ps((float*)(a2))),_mm_and_ps(a0,a1)));
    result += 4;
    a2++;
  }
}
void inWHEREvss(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_or_ps(_mm_andnot_ps(_mm_load_ps((float*)(a0)),a2),_mm_and_ps(_mm_load_ps((float*)(a0)),a1)));
    result += 4;
    a0++;
  }
}
void inMADvvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_mul_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))),_mm_load_ps((float*)(a2))));
    result += 4;
    a0++,a1++,a2++;
  }
}
void inMADvvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_mul_ps(_mm_load_ps((float*)(a0)),_mm_load_ps((float*)(a1))),a2));
    result += 4;
    a0++,a1++;
  }
}
void inMADvsv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_mul_ps(_mm_load_ps((float*)(a0)),a1),_mm_load_ps((float*)(a2))));
    result += 4;
    a0++,a2++;
  }
}
void inMADsvv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_mul_ps(a0,_mm_load_ps((float*)(a1))),_mm_load_ps((float*)(a2))));
    result += 4;
    a1++,a2++;
  }
}
void inMADsvs(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_mul_ps(a0,_mm_load_ps((float*)(a1))),a2));
    result += 4;
    a1++;
  }
}
void inMADssv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_mul_ps(a0,a1),_mm_load_ps((float*)(a2))));
    result += 4;
    a2++;
  }
}
void inMADvss(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_mul_ps(_mm_load_ps((float*)(a0)),a1),a2));
    result += 4;
    a0++;
  }
}
void inLERPvvv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_load_ps((float*)(a1)),_mm_mul_ps(_mm_load_ps((float*)(a0)),_mm_sub_ps(_mm_load_ps((float*)(a2)),_mm_load_ps((float*)(a1))))));
    result += 4;
    a0++,a1++,a2++;
  }
}
void inLERPvvs(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 *a1 = (__m128*)a[1];
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_load_ps((float*)(a1)),_mm_mul_ps(_mm_load_ps((float*)(a0)),_mm_sub_ps(a2,_mm_load_ps((float*)(a1))))));
    result += 4;
    a0++,a1++;
  }
}
void inLERPvsv(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(a1,_mm_mul_ps(_mm_load_ps((float*)(a0)),_mm_sub_ps(_mm_load_ps((float*)(a2)),a1))));
    result += 4;
    a0++,a2++;
  }
}
void inLERPsvv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_load_ps((float*)(a1)),_mm_mul_ps(a0,_mm_sub_ps(_mm_load_ps((float*)(a2)),_mm_load_ps((float*)(a1))))));
    result += 4;
    a1++,a2++;
  }
}
void inLERPsvs(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 *a1 = (__m128*)a[1];
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(_mm_load_ps((float*)(a1)),_mm_mul_ps(a0,_mm_sub_ps(a2,_mm_load_ps((float*)(a1))))));
    result += 4;
    a1++;
  }
}
void inLERPssv(float *result, float **a, int l)
{
  int i;
  register __m128 a0 = _mm_setr_ps(*a[0],*a[0],*a[0],*a[0]);
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 *a2 = (__m128*)a[2];
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(a1,_mm_mul_ps(a0,_mm_sub_ps(_mm_load_ps((float*)(a2)),a1))));
    result += 4;
    a2++;
  }
}
void inLERPvss(float *result, float **a, int l)
{
  int i;
  register __m128 *a0 = (__m128*)a[0];
  register __m128 a1 = _mm_setr_ps(*a[1],*a[1],*a[1],*a[1]);
  register __m128 a2 = _mm_setr_ps(*a[2],*a[2],*a[2],*a[2]);
  for (i = ((l+3)>>2); i; i--) {
    _mm_store_ps(result, _mm_add_ps(a1,_mm_mul_ps(_mm_load_ps((float*)(a0)),_mm_sub_ps(a2,a1))));
    result += 4;
    a0++;
  }
}
