def genOp(name, props, opcode, restype = 'float'):
    arity = len(props)
    yield "void in%s%s(%s *result, float **a, int l)" % (name, "".join(props), restype)
    yield "{"
    yield "  int i;"
    a = [ None ] * len(props)
    post = []
    for i,p in enumerate(props):
        if p == 's':
            yield "  register __m128 a%d = _mm_setr_ps(*a[%d],*a[%d],*a[%d],*a[%d]);" % (i,i,i,i,i)
            a[i] = "a%d" % i
        else:
            yield "  register __m128 *a%d = (__m128*)a[%d];" % (i,i)
            a[i] = "_mm_load_ps((float*)(a%d))" % i
            post.append("a%d++" % i)
    yield "  for (i = ((l+3)>>2); i; i--) {"
    res = opcode
    for i,ia in enumerate(a):
      res = res.replace('$'+str(i), ia)
    if restype == 'float':
        yield "    _mm_store_ps(result, %s);" % res
        yield "    result += 4;"
    else:
        yield "    *result++ = %s;" % res
    yield "    %s;" % ",".join(post)
    yield "  }"
    yield "}"

bincombos = [ ['v','v'], ['v','s'], ['s','v'] ]
terncombos = [ ['v']+c for c in bincombos ] + [ ['s']+c for c in bincombos ] + [['v','s','s']]

def unary(name, op, restype='float'):
    for l in genOp(name, ['v'], op, restype):
      print l

def binary(name, op):
    for c in bincombos:
        for l in genOp(name, c, op):
          print l

def ternary(name, op):
    for c in terncombos:
        for l in genOp(name, c, op):
          print l

def k(n):
  return '(__m128)_mm_set_epi32(%s,%s,%s,%s)' % (n,n,n,n)

unary('SQRT', '_mm_sqrt_ps($0)')
unary('NEG', '_mm_xor_ps($0,%s)' % k('0x80000000'))
unary('NOT', '_mm_xor_ps($0,%s)' % k('0xffffffff'))
unary('ABS', '_mm_and_ps($0,%s)' % k('0x7fffffff'))
unary('INT', '(__m128)_mm_cvtps_epi32($0)')
unary('FLOOR', '(__m128)_mm_cvtepi32_ps(_mm_cvtps_epi32($0))')
unary('CHAR', '_mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi16(_mm_cvtps_epi32($0),(__m128i)%s),(__m128i)%s))' % (k('0'), k('0')), 'int')

binary('MIN', '_mm_min_ps($0,$1)')
binary('MAX', '_mm_max_ps($0,$1)')
binary('ADD', '_mm_add_ps($0,$1)')
binary('SUB', '_mm_sub_ps($0,$1)')
binary('MUL', '_mm_mul_ps($0,$1)')
binary('DIV', '_mm_div_ps($0,$1)')
binary('AND', '_mm_and_ps($0,$1)')
binary('OR', '_mm_or_ps($0,$1)')
binary('XOR', '_mm_xor_ps($0,$1)')
binary('EQ', '_mm_cmpeq_ps($0,$1)')
binary('NE', '_mm_cmpneq_ps($0,$1)')
binary('LT', '_mm_cmplt_ps($0,$1)')
binary('GE', '_mm_cmpnlt_ps($0,$1)')
binary('LE', '_mm_cmple_ps($0,$1)')
binary('GT', '_mm_cmpnle_ps($0,$1)')

ternary('WHERE', '_mm_or_ps(_mm_andnot_ps($0,$2),_mm_and_ps($0,$1))')
ternary('MAD', '_mm_add_ps(_mm_mul_ps($0,$1),$2)')
# (a + t * (b - a))
# $1 + $0 * ($2 - $1)
ternary('LERP', '_mm_add_ps($1,_mm_mul_ps($0,_mm_sub_ps($2,$1)))')
