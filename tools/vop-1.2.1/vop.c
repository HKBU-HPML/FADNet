/*
* Copyright (C) 2007 James Bowman, All rights reserved.
* Written by James Bowman
*/

/* headers */
#include <Python.h>

#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#endif

#include <assert.h>

#include <math.h>

#define TRON 0

#if TRON
#define TRACE(X)  X
#else
#define TRACE(X)
#endif

/* Module globals */
static PyObject *vop_error = NULL;

static long long odometer;

/* Vop type */
staticforward PyTypeObject Vop_Type;

/* Vop object */
typedef struct {
    PyObject_HEAD
    long length;
    long truelength;
    char *raw_ptr;
    float *data;
} Vop;

#define Vop_Check(v)  ((v)->ob_type == &Vop_Type)
#define Vop_length(v)  (((Vop *)(v))->length)
#define Vop_truelength(v)  (((Vop *)(v))->truelength)

/* constructor */
static PyObject *
Vop_NEW(long initial_length)
{
    Vop *object;

    long rounded_length = (initial_length + 3) & ~3;
    
    object = PyObject_NEW(Vop, &Vop_Type);
    //printf("NEW %p\n", object);
    if (object != NULL) {
        object->truelength = initial_length;
        object->length = rounded_length;
    }
    char *raw_ptr = malloc(4 * rounded_length + 16);
    object->raw_ptr = raw_ptr;
    object->data = (void*)(((size_t)raw_ptr + 15) & ~15);

    TRACE(printf("%x\n", (int)object->data));
    if (object->data == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    return (PyObject *)object;
}

static void
Vop_dealloc(PyObject *self)
{
    TRACE(printf("Before free line=%d\n", __LINE__));
    free(((Vop *)self)->raw_ptr);
    TRACE(printf("After free line=%d\n", __LINE__));
    // PyMem_DEL(self);
    TRACE(printf("After del line=%d\n", __LINE__));
    //printf("FREE %p\n", self);
    PyObject_Del(self);
}

/* Numeric emulation routines */

/* In this case, it is easier to write a coercion routines than to
 * all the little add, subtract, multiply, etc. routines.
 */

#include <xmmintrin.h>
#include "generated.i"

static PyObject *Vop_Neg(PyObject *self)
{
    int l = Vop_truelength(self);
    Vop *s = (Vop*)self;
    Vop *r = (Vop*)Vop_NEW(l);
#if 0
    int i;
    for (i = 0; i < l; i++) {
        //printf("i=%d l=%d\n", i, l);
        r->data[i] = -s->data[i];
    }
#else
    float *arrays[1];
    arrays[0] = s->data;
    inNEGv(r->data, arrays, l);
#endif
    odometer += l;
    return (PyObject*)r;
}

static PyObject *Vop_Abs(PyObject *self)
{
    int l = Vop_truelength(self);
    Vop *s = (Vop*)self;
    Vop *r = (Vop*)Vop_NEW(l);
#if 0
    int i;
    for (i = 0; i < l; i++) {
        //printf("i=%d l=%d\n", i, l);
        r->data[i] = fabs(s->data[i]);
    }
#else
    float *arrays[1];
    arrays[0] = s->data;
    inABSv(r->data, arrays, l);
#endif
    odometer += l;
    return (PyObject*)r;
}

typedef void (*utility)(float *r, float **a, int l);

static float toconst(PyObject *p)
{
    if (PyBool_Check(p)) {
        long r;

        if (p == Py_False)
            r = 0;
        else
            r = ~0;
        return *(float*)&r;
    } else {
        return PyFloat_AsDouble(p);
    }
}

static PyObject *Vop_inplaceAdd(PyObject *self, PyObject *other)
{
    int l = Vop_truelength(self);
    Vop *s = (Vop*)self;
    int which = 0;
    float f;
    float *arrays[2];

    utility funcs[2] = {
        inADDvv,
        inADDvs
    };

    arrays[0] = ((Vop*)self)->data;
    if (Vop_Check(other)) {
        arrays[1] = ((Vop*)other)->data;
    } else {
        f = toconst(other);
        arrays[1] = &f;
        which += 1;
    }
    (funcs[which])(s->data, arrays, l);
    Py_INCREF(self);

    return self;
}

static PyObject *Vop_Binary(PyObject *self, PyObject *other, utility func[3], int arith)
{
    int l;
    float f[2];
    int which = 0;

    float *arrays[2];
    Vop *r;

    if (Vop_Check(self) && Vop_Check(other)) {
        if (Vop_truelength(self) != Vop_truelength(other)) {
            PyErr_BadArgument();
            return NULL;
        }
    }

    if (Vop_Check(self)) {
        l = Vop_truelength(self);
        arrays[0] = ((Vop*)self)->data;
    } else {
        f[0] = toconst(self);
        arrays[0] = &f[0];
        which += 1;
    }

    if (Vop_Check(other)) {
        l = Vop_truelength(other);
        arrays[1] = ((Vop*)other)->data;
    } else {
        f[1] = toconst(other);
        arrays[1] = &f[1];
        which += 2;
    }
    r = (Vop*)Vop_NEW(l);

    if (r == NULL) {
        return NULL;
    }

#if TRON
    printf("%c%c which=%d\n", Vop_Check(self) ? 'v' : 's', Vop_Check(other) ? 'v' : 's', which);
    printf("%p %p %p %d\n", r->data, arrays[0], arrays[1], l);
#endif
    assert(r->data != NULL);

    (func[which])(r->data, arrays, l);

    if (0 && arith) {
        int i;
        long and = ~0;
        long or = 0;
        for (i = 0; i < l; i++) {
            and &= *(long*)&r->data[i];
            or |= *(long*)&r->data[i];
        }
        if (and == or) {
            // printf("Elem zero is %f\n", r->data[0]);
            PyObject *rv = PyFloat_FromDouble(r->data[0]);
            free(r->data);
            PyObject_Del(r);
            return rv;
        }
        // printf("Length %d, allsame %d\n", l, allsame);
    }

    odometer += l;
    TRACE(printf("Back from asm at line %d\n", __LINE__));

    return (PyObject*)r;
}

#define BIN(OP, arith)                                             \
static PyObject *Vop_##OP(PyObject *self, PyObject *other)  \
{                                                           \
    utility funcs[3] = {                                    \
        in##OP##vv,                                         \
        in##OP##sv,                                         \
        in##OP##vs                                          \
    };                                                      \
    TRACE(printf("Vop_%s\n", #OP));                            \
    return Vop_Binary(self, other, funcs, arith);                  \
}

BIN(ADD, 1)
BIN(MUL, 1)
BIN(SUB, 1)

BIN(MAX, 1)
BIN(MIN, 1)

BIN(AND, 0)
BIN(OR, 0)
BIN(XOR, 0)

BIN(EQ, 0)
BIN(NE, 0)
BIN(LT, 0)
BIN(GT, 0)
BIN(LE, 0)
BIN(GE, 0)


#if 0
BIN(DIV)
#else
static void trickDIVvs(float *r, float **aa, int l)
{
    float *newaa[2];
    float recip;

    newaa[0] = aa[0];
    recip = 1.0 / *(aa[1]);
    newaa[1] = &recip;

    inMULvs(r, newaa, l);
}

static PyObject *Vop_DIV(PyObject *self, PyObject *other)
{
    utility funcs[3] = {                                    
        inDIVvv,                                        
        inDIVsv,                                       
        trickDIVvs                                       
    };                                                  
    TRACE(printf("Vop_%s\n", "DIV"));                    
    return Vop_Binary(self, other, funcs, 1);            
}
#endif

static PyObject *Vop_Invert(PyObject *self)
{
    int l = Vop_truelength(self);
    Vop *s = (Vop*)self;
    Vop *r = (Vop*)Vop_NEW(l);

    float *arrays[1];
    arrays[0] = s->data;
    inNOTv(r->data, arrays, l);
    odometer += l;

    return (PyObject*)r;
}

static PyNumberMethods Vop_as_number = {
  Vop_ADD,                /* nb_add */
  Vop_SUB,                /* nb_subtract */
  Vop_MUL,     /* nb_multiply */
  Vop_DIV,          /* nb_divide */
  0,                /* nb_remainder */
  0,                /* nb_divmod */
  0,                /* nb_power */
  Vop_Neg,      /* nb_negative */
  0,      /* nb_positive */
  Vop_Abs,      /* nb_absolute */
  0,  /* nb_nonzero */
  Vop_Invert,   /* nb_invert */
  0,                /* nb_lshift */
  0,                /* nb_rshift */
  Vop_AND,                /* nb_and */
  Vop_XOR,                /* nb_xor */
  Vop_OR,                /* nb_or */
  0,   /* nb_coerce */
  0,      /* nb_int */
  0,     /* nb_long */
  0,     /* nb_float */
  0,      /* nb_oct */
  0,      /* nb_hex */
  Vop_inplaceAdd,
};

/* type methods */

static PyObject *
Vop_tostring(PyObject *self, PyObject *args)
{
    PyObject *result = NULL;

    Vop *v = (Vop*)self;
    char *s = malloc(v->length + 16);
    int i;
#if 0
    for (i = 0; i < v->length; i++) {
        s[i] = (char)(v->data[i]);
    }
#else
    float *arrays[2];
    arrays[0] = v->data;
    TRACE(printf("CHAR with length %d\n", v->truelength));
    inCHARv((int*)s, arrays, v->truelength);
#endif
    result = PyString_FromStringAndSize(s, v->truelength);
    free(s);
    return result;
}

static PyObject *
Vop_tohalf(PyObject *self, PyObject *args)
{
    assert(0);
#if 0
    PyObject *result = NULL;

    Vop *v = (Vop*)self;
    char *s = malloc(2 * v->length + 16);
    int i;
#if 0
    for (i = 0; i < v->length; i++) {
        s[i] = (char)(v->data[i]);
    }
#else
    float *arrays[2];
    arrays[0] = v->data;
    TRACE(printf("HALF with length %d\n", v->truelength));
    doHALF(s, arrays, v->truelength);
#endif
    result = PyString_FromStringAndSize(s, v->truelength * 2);
    odometer += v->truelength;
    free(s);
    return result;
#endif
}

static PyObject *
Vop_toraw(PyObject *self, PyObject *args)
{
    Vop *v = (Vop*)self;
    PyObject *result = NULL;

    result = PyString_FromStringAndSize((char*)v->data, v->truelength * 4);
    return result;
}

static PyObject *
Vop_tolist(PyObject *self, PyObject *args)
{
    PyObject *result = NULL;
    int i;
    Vop *v = (Vop*)self;

    result = PyList_New(v->truelength);
    for (i = 0; i < v->truelength; i++) {
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(v->data[i]));
    }
    return result;
}

static PyObject *
Vop_sum(PyObject *self, PyObject *args)
{
    PyObject *result = NULL;

    Vop *v = (Vop*)self;
    int i;
    float sum = 0.0;
    for (i = 0; i < v->truelength; i++) {
        sum += v->data[i];
    }
    return PyFloat_FromDouble(sum);
}

/* Method table */
static PyMethodDef Vop_methods[] = {
  {"tostring", Vop_tostring, METH_VARARGS},
  {"tolist", Vop_tolist, METH_VARARGS},
  {"toraw", Vop_toraw, METH_VARARGS},
  {"tohalf", Vop_tohalf, METH_VARARGS},
  {"sum", Vop_sum, METH_VARARGS},
  {NULL, NULL},
};

/* Callback routines */

static PyObject * Vop_GetAttr(PyObject *self, char*attrname)
{
    return Py_FindMethod(Vop_methods, self, attrname);
}

static PyObject *Vop_Repr(PyObject *self)
{
    PyObject *result = NULL;
    char buf[50];
    sprintf(buf, "<vop of length %ld>", Vop_truelength(self));
    return PyString_FromString(buf);
}

static PyObject *Vop_compare(PyObject *a, PyObject *b, int opid)
{
    switch (opid) {
    case Py_EQ:
        return Vop_EQ(a, b);
    case Py_LT:
        return Vop_LT(a, b);
    case Py_LE:
        return Vop_LE(a, b);
    case Py_GT:
        return Vop_GT(a, b);
    case Py_GE:
        return Vop_GE(a, b);
    case Py_NE:
        return Vop_NE(a, b);
    }
    return a;
}

int Vop_LENGTH(PyObject *v)
{
    return Vop_truelength(v);
}

PyObject *Vop_GetItem(PyObject *v, Py_ssize_t i)
{
    if (i < Vop_truelength(v)) {
        Vop *vv = (Vop*)v;
        return PyFloat_FromDouble(vv->data[i]);
    } else {
        PyErr_SetNone(PyExc_IndexError);
        return NULL;
    }
    return PyString_FromString("hello");
}

static PySequenceMethods Vop_as_sequence = {
    &Vop_LENGTH,
    NULL,
    NULL,
    &Vop_GetItem
};
    

/* Type definition */
/* remember the forward declaration above, this is the real definition */
static PyTypeObject Vop_Type = {
    PyObject_HEAD_INIT(&PyType_Type)
    0,
    "vop",
    sizeof(Vop),
    0,
    (destructor)Vop_dealloc,
    0,
    (getattrfunc)Vop_GetAttr,
    0,
    0,
    (reprfunc)Vop_Repr,
    &Vop_as_number,
    &Vop_as_sequence,
    0,

    0,
    0,
    0,
    0,
    0,
    
    0,
    
    // in place seems to buy nothing.
    Py_TPFLAGS_CHECKTYPES | Py_TPFLAGS_HAVE_RICHCOMPARE, // | Py_TPFLAGS_HAVE_INPLACEOPS,

    0,
    0,
    0,
    &Vop_compare

    /* the rest are NULLs */
};

/* Module functions */

PyObject *arange(PyObject *self, PyObject *args)
{
    PyObject *result = NULL;
    float f;
    int i;

    if (PyArg_ParseTuple(args, "f", &f)) {
        result = Vop_NEW((int)f);
        for (i = 0; i < (int)f; i++) {
            ((Vop*)result)->data[i] = (float)i;
        }
    } /* otherwise there is an error,
       * the exception already raised by PyArg_ParseTuple, and NULL is
       * returned.
       */

    return result;
}


PyObject *array(PyObject *self, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *a, *sf;
    if (!PyArg_ParseTuple(args, "O", &a)) return 0;
    if (PyString_Check(a)) {
        char *s;
        int l = 1;
        int rv;
        int i;
        unsigned char *us;

        rv = PyString_AsStringAndSize(a, &s, &l);
        TRACE(printf("array from string (length %d)\n", l));
        assert(rv == 0);

        us = (unsigned char *)s;
        result = Vop_NEW(l);
        for (i = 0; i < l; i++) {
            ((Vop*)result)->data[i] = (float)us[i];
        }
    } else if ((sf = PySequence_Fast(a, "must be iterable")) != NULL) {
        TRACE(printf("array from list %d\n", l));
        Py_ssize_t l = PySequence_Fast_GET_SIZE(sf);
        Py_ssize_t i;
        result = Vop_NEW(l);
        for (i = 0; i < l; i++) {
            ((Vop*)result)->data[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(sf, i));
        }
        Py_DECREF(sf);
    } else {
        assert(0);
    }
    TRACE(printf("array %p\n", result));

    return result;
}

PyObject *fromstring(PyObject *self, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *a;

    char *s;
    int l = 1;
    int rv;
    int i;
    float *fs;

    if (!PyArg_ParseTuple(args, "O", &a)) return 0;
    rv = PyString_AsStringAndSize(a, &s, &l);
    assert(rv == 0);
    fs = (float *)s;

    l >>= 2;
    TRACE(printf("fromstring %d\n",l));

    result = Vop_NEW(l);
    for (i = 0; i < l; i++) {
        ((Vop*)result)->data[i] = fs[i];
    }

    return result;
}

PyObject *mysqrt(PyObject *self, PyObject *args)
{
    self = PyTuple_GetItem(args, 0);

    if (Vop_Check(self)) {
        int l = Vop_truelength(self);
        Vop *s = (Vop*)self;
        Vop *r = (Vop*)Vop_NEW(l);

        float *arrays[1];
        arrays[0] = s->data;
        TRACE(printf("r->data=%p, arrays[0]=%p, l=%d\n", r->data, arrays[0], l));
        inSQRTv(r->data, arrays, l);
        odometer += l;

        TRACE(printf("after\n"));
        return (PyObject*)r;
    } else {
        return PyFloat_FromDouble(sqrt(PyFloat_AsDouble(self)));
    }
}

PyObject *myfloor(PyObject *self, PyObject *args)
{
    self = PyTuple_GetItem(args, 0);
    if (Vop_Check(self)) {
        int l = Vop_truelength(self);
        Vop *s = (Vop*)self;
        Vop *r = (Vop*)Vop_NEW(l);

        float *arrays[1];
        arrays[0] = s->data;
        _mm_setcsr(0x3f80);
        inFLOORv(r->data, arrays, l);
        odometer += l;

        return (PyObject*)r;
    } else {
        return PyFloat_FromDouble(floor(PyFloat_AsDouble(self)));
    }
}

PyObject *myacos(PyObject *self, PyObject *args)
{
    self = PyTuple_GetItem(args, 0);
    if (Vop_Check(self)) {
        int l = Vop_truelength(self);
        Vop *s = (Vop*)self;
        Vop *r = (Vop*)Vop_NEW(l);

        int i;
        for (i = 0; i < l; i++)
            r->data[i] = acos(s->data[i]);
        odometer += l;

        return (PyObject*)r;
    } else {
        return PyFloat_FromDouble(acos(PyFloat_AsDouble(self)));
    }
}

PyObject *mycos(PyObject *self, PyObject *args)
{
    self = PyTuple_GetItem(args, 0);
    if (Vop_Check(self)) {
        int l = Vop_truelength(self);
        Vop *s = (Vop*)self;
        Vop *r = (Vop*)Vop_NEW(l);

        int i;
        for (i = 0; i < l; i++)
            r->data[i] = cos(s->data[i]);
        odometer += l;

        return (PyObject*)r;
    } else {
        return PyFloat_FromDouble(cos(PyFloat_AsDouble(self)));
    }
}

PyObject *mysin(PyObject *self, PyObject *args)
{
    self = PyTuple_GetItem(args, 0);
    if (Vop_Check(self)) {
        int l = Vop_truelength(self);
        Vop *s = (Vop*)self;
        Vop *r = (Vop*)Vop_NEW(l);

        int i;
        for (i = 0; i < l; i++)
            r->data[i] = sin(s->data[i]);
        odometer += l;

        return (PyObject*)r;
    } else {
        return PyFloat_FromDouble(sin(PyFloat_AsDouble(self)));
    }
}

PyObject *myexp(PyObject *self, PyObject *args)
{
    self = PyTuple_GetItem(args, 0);
    if (Vop_Check(self)) {
        int l = Vop_truelength(self);
        Vop *s = (Vop*)self;
        Vop *r = (Vop*)Vop_NEW(l);

        int i;
        for (i = 0; i < l; i++)
            r->data[i] = exp(s->data[i]);
        odometer += l;

        return (PyObject*)r;
    } else {
        return PyFloat_FromDouble(exp(PyFloat_AsDouble(self)));
    }
}


// No measurable speed benefit from the asm versions of these.

PyObject *sometrue(PyObject *self, PyObject *args)
{
    self = PyTuple_GetItem(args, 0);
    if (Vop_Check(self)) {
        Vop *s = (Vop*)self;

#if 0
        long out = doSOMETRUE(s->data, s->truelength);
#else
        long out = 0;
        int i;
        for (i = 0; i < s->truelength; i++)
            out |= *(long*)&s->data[i];
#endif
        odometer += s->truelength;

        return PyBool_FromLong(out);
    } else {
        if (self == Py_False)
            return PyBool_FromLong(0);
        else if ((PyFloat_Check(self) && PyFloat_AsDouble(self) == 0.0))
            return PyBool_FromLong(0);
        else
            return PyBool_FromLong(1);
    }
}


PyObject *alltrue(PyObject *self, PyObject *args)
{
    self = PyTuple_GetItem(args, 0);
    if (Vop_Check(self)) {
        Vop *s = (Vop*)self;

#if 0
        long out = doALLTRUE(s->data, s->truelength);
#else
        long out = ~0;
        int i;
        for (i = 0; i < s->truelength; i++)
            out &= *(long*)&s->data[i];
#endif
        odometer += s->truelength;

        return PyBool_FromLong(out);
    } else {
        if (self == Py_False)
            return PyBool_FromLong(0);
        else if ((PyFloat_Check(self) && PyFloat_AsDouble(self) == 0.0))
            return PyBool_FromLong(0);
        else
            return PyBool_FromLong(1);
    }
}

PyObject *minimum(PyObject *self, PyObject *args)
{
    PyObject *a = PyTuple_GetItem(args, 0);
    PyObject *b = PyTuple_GetItem(args, 1);

    if (!Vop_Check(a) && !Vop_Check(b)) {
        float fa = PyFloat_AsDouble(a);
        float fb = PyFloat_AsDouble(b);

        return PyFloat_FromDouble(fa < fb ? fa : fb);
    } else {
        odometer += Vop_truelength(a);
        return Vop_MIN(a, b);
    }
}

PyObject *maximum(PyObject *self, PyObject *args)
{
    PyObject *a = PyTuple_GetItem(args, 0);
    PyObject *b = PyTuple_GetItem(args, 1);

    if (!Vop_Check(a) && !Vop_Check(b)) {
        float fa = PyFloat_AsDouble(a);
        float fb = PyFloat_AsDouble(b);

        return PyFloat_FromDouble(fa < fb ? fb : fa);
    } else {
        odometer += Vop_truelength(a);
        return Vop_MAX(a, b);
    }
}

PyObject *compress(PyObject *self, PyObject *args)
{
    PyObject *mask = PyTuple_GetItem(args, 0);
    PyObject *vals = PyTuple_GetItem(args, 1);

    if (!Vop_Check(vals)) {
        Py_INCREF(vals);
        return vals;
    } else {
        Vop *m = (Vop*)mask;
        Vop *v = (Vop*)vals;
        Vop *r;
        int i, l;

        if (Vop_truelength(mask) != Vop_truelength(vals)) {
            PyErr_BadArgument();
            return NULL;
        }
        l = 0;
        for (i = 0; i < m->truelength; i++)
            if (m->data[i])
                l++;
        r = (Vop*)Vop_NEW(l);

        l = 0;
        for (i = 0; i < m->truelength; i++)
            if (m->data[i])
                r->data[l++] = v->data[i];

        odometer += l;

        assert(l <= r->truelength);
        return (PyObject*)r;
    }
}

PyObject *expand(PyObject *self, PyObject *args)
{
    PyObject *mask = PyTuple_GetItem(args, 0);
    PyObject *vals = PyTuple_GetItem(args, 1);
    PyObject *defl = PyTuple_GetItem(args, 2);

    if (!Vop_Check(vals)) {
        Py_INCREF(vals);
        return vals;
    } else {
        Vop *m = (Vop*)mask;
        Vop *v = (Vop*)vals;
        float d = PyFloat_AsDouble(defl);

        Vop *r = (Vop*)Vop_NEW(m->truelength);
        int i, l = 0;
        for (i = 0; i < m->truelength; i++)
            if (m->data[i])
                r->data[i] = v->data[l++];
            else
                r->data[i] = d;

        odometer += l;

        return (PyObject*)r;
    }
}

PyObject *replicate(PyObject *self, PyObject *args)
{
    PyObject *base = PyTuple_GetItem(args, 0);
    PyObject *count = PyTuple_GetItem(args, 1);

    if (!Vop_Check(base)) {
        assert(0);
    } else {
        Vop *m = (Vop*)base;
        int c = PyInt_AsLong(count);

        Vop *r = (Vop*)Vop_NEW(m->truelength * c);
        int i, j, l;
        l = 0;
        for (i = 0; i < m->truelength; i++) {
            for (j = 0; j < c; j++)
                r->data[l++] = m->data[i];
        }
        TRACE(printf("Replicated %d * %d -> %d\n", m->truelength, c, l));

        return (PyObject*)r;
    }
}

PyObject *duplicate(PyObject *self, PyObject *args)
{
    PyObject *base = PyTuple_GetItem(args, 0);
    PyObject *count = PyTuple_GetItem(args, 1);

    if (!Vop_Check(base)) {
        assert(0);
    } else {
        Vop *m = (Vop*)base;
        int c = PyInt_AsLong(count);

        Vop *r = (Vop*)Vop_NEW(m->truelength * c);
        int i, j, l;
        l = 0;
        for (j = 0; j < c; j++)
            for (i = 0; i < m->truelength; i++) {
                r->data[l++] = m->data[i];
        }
        TRACE(printf("Duplicated %d * %d -> %d\n", m->truelength, c, l));

        return (PyObject*)r;
    }
}

static void indirectF(float *dst, float *src, int *indices, size_t L)
{
    int i;
    for (i = 0; i < L; i++)
      dst[i] = src[indices[i]];
}

static void indirectB(float *dst, unsigned char *src, int *indices, size_t L)
{
    int i;
    for (i = 0; i < L; i++) {
      dst[i] = (float)src[indices[i]];
    }
}

PyObject *take(PyObject *self, PyObject *args)
{
    PyObject *data = PyTuple_GetItem(args, 0);
    PyObject *idxs = PyTuple_GetItem(args, 1);

    if (Vop_Check(data) && Vop_Check(idxs)) {
        Vop *d = (Vop*)data;
        Vop *i = (Vop*)idxs;

        Vop *r = (Vop*)Vop_NEW(i->truelength);

        float *arrays[2];
        arrays[0] = i->data;

        inINTv(r->data, arrays, i->truelength);

        int *indices = (int*)r->data;
        float *dst = r->data;
        float *src = d->data;
        indirectF(dst, src, indices, i->truelength);
        odometer += i->truelength;
        return (PyObject*)r;
    } else if (!Vop_Check(data)) {
        return PyFloat_FromDouble(PyFloat_AsDouble(data));
    } else {
        // data vector, idxs is scalar
        Vop *d = (Vop*)data;
        return PyFloat_FromDouble(d->data[(int)PyFloat_AsDouble(idxs)]);
    }
}

PyObject *take2(PyObject *self, PyObject *args)
{
    assert(0);
#if 0
    PyObject *data = PyTuple_GetItem(args, 0);
    PyObject *idxXs = PyTuple_GetItem(args, 1);
    PyObject *idxYs = PyTuple_GetItem(args, 2);
    PyObject *stride = PyTuple_GetItem(args, 3);

    if (Vop_Check(data) && Vop_Check(idxXs) && Vop_Check(idxYs)) {
        Vop *d = (Vop*)data;
        Vop *i = (Vop*)idxXs;

        Vop *r = (Vop*)Vop_NEW(i->truelength);

        float *arrays[4];
        arrays[0] = d->data;
        arrays[1] = ((Vop*)idxXs)->data;
        arrays[2] = ((Vop*)idxYs)->data;
        long istride = PyInt_AsLong(stride);
        arrays[3] = *(float**)&istride;

        doTAKE2(r->data, arrays, i->truelength);
        odometer += i->truelength;
        return (PyObject*)r;
    } else {
        assert(0);
    }
#endif
}


PyObject *takeB(PyObject *self, PyObject *args)
{
    PyObject *data = PyTuple_GetItem(args, 0);
    PyObject *idxs = PyTuple_GetItem(args, 1);

    if (Vop_Check(idxs)) {
        Vop *d = (Vop*)data;
        Vop *i = (Vop*)idxs;

        Vop *r = (Vop*)Vop_NEW(i->truelength);

        float *arrays[4];
        arrays[0] = i->data;

        inINTv(r->data, arrays, i->truelength);

        int *indices = (int*)r->data;
        float *dst = r->data;
        unsigned char *src = (unsigned char*)d->data;
        indirectB(dst, src, indices, i->truelength);

        odometer += i->truelength;
        return (PyObject*)r;
    } else {
        // data vector, idxs is scalar
        Vop *d = (Vop*)data;
        return PyFloat_FromDouble(PyString_AsString(data)[(int)PyFloat_AsDouble(idxs)]);
    }
}

PyObject *take2B(PyObject *self, PyObject *args)
{
    assert(0);
#if 0
    PyObject *data = PyTuple_GetItem(args, 0);
    PyObject *idxXs = PyTuple_GetItem(args, 1);
    PyObject *idxYs = PyTuple_GetItem(args, 2);
    PyObject *stride = PyTuple_GetItem(args, 3);

    if (Vop_Check(idxXs) && Vop_Check(idxYs)) {
        Vop *d = (Vop*)data;
        Vop *i = (Vop*)idxXs;

        Vop *r = (Vop*)Vop_NEW(i->truelength);

        float *arrays[4];
        arrays[0] = (float*)PyString_AsString(data);
        arrays[1] = ((Vop*)idxXs)->data;
        arrays[2] = ((Vop*)idxYs)->data;
        long istride = PyInt_AsLong(stride);
        arrays[3] = *(float**)&istride;

        doTAKE2B(r->data, arrays, i->truelength);
        odometer += i->truelength;
        return (PyObject*)r;
    } else {
        PyErr_SetString(PyExc_TypeError, "expecting number");
        return 0;
    }
#endif
}


PyObject *where(PyObject *self, PyObject *args)
{
    int i, l;
    float *arrays[3], f[3];
    int which = 0;
    utility funcs[7] = {
        inWHEREvvv,
        inWHEREvvs,
        inWHEREvsv,
        inWHEREvss,
        inWHEREsvv,
        inWHEREsvs,
        inWHEREssv
    };

    int otherl = -1;
    for (i = 0; i < 3; i++) {
        PyObject *it = PyTuple_GetItem(args, i);
        if (Vop_Check(it)) {
            l = Vop_truelength(it);
            if (otherl != -1) {
                if (l != otherl) {
                    PyErr_BadArgument();
                    return NULL;
                }
            }
            otherl = l;
            arrays[i] = ((Vop*)it)->data;
        } else {
            f[i] = PyFloat_AsDouble(it);
            arrays[i] = &f[i];
            which += (4 >> i);
        }
    }
    if (which < 7) {
        Vop *r = (Vop*)Vop_NEW(l);

        odometer += l;
        if (which == 0) {
            int i = l - 1;
            TRACE(printf("%f %f %f\n", *(arrays[0]), *(arrays[1]), *(arrays[2])));
            TRACE(printf("%f %f %f\n", (arrays[0])[i], (arrays[1])[i], (arrays[2])[i]));
        }
        TRACE(printf("calling inWHERE[%d] result %p src %p,%p,%p\n", which, r->data, arrays[0], arrays[1], arrays[2]));
        (funcs[which])(r->data, arrays, l);
        
        return (PyObject *)r;
    } else {
        PyObject *pred = PyTuple_GetItem(args, 0);
        double t = PyFloat_AsDouble(PyTuple_GetItem(args, 1));
        double f = PyFloat_AsDouble(PyTuple_GetItem(args, 2));
        return PyFloat_FromDouble((pred == Py_False) ? f : t);
    }
}

/*
 * Implement the ternary operator MAD (multiply-add)
 * mad(a, b, c) returns (a*b)+c
 */

PyObject *mad(PyObject *self, PyObject *args)
{
    int i, length = -1;
    float *arrays[3], f[3];
    int which = 0;
    utility funcs[7] = {
        inMADvvv,
        inMADvvs,
        inMADvsv,
        inMADvss,
        inMADsvv,
        inMADsvs,
        inMADssv
    };

    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "too few arguments");
        return NULL;
    }

    // decide if each argument is a scalar or a vector, compute 'which' and
    // set up 'arrays' 

    for (i = 0; i < 3; i++) {
        PyObject *it = PyTuple_GetItem(args, i);
        if (Vop_Check(it)) {
            // first vector, set the length.  Subseequent vectors,
            // want to check that they have matching length
            if (length == -1)
                length = Vop_truelength(it);
            else if (length != Vop_truelength(it)) {
                PyErr_SetString(PyExc_TypeError, "mismatching lengths");
                return NULL;
            }
            arrays[i] = ((Vop*)it)->data;
        } else if (PyFloat_Check(it)) {
            f[i] = PyFloat_AsDouble(it);
            arrays[i] = &f[i];
            which += (4 >> i);
        } else {
            PyErr_SetString(PyExc_TypeError, "expecting vector or float");
            return NULL;
        }
    }
    if (which < 7) {
        Vop *r = (Vop*)Vop_NEW(length);

        odometer += 3 * length;
        (funcs[which])(r->data, arrays, length);
        
        return (PyObject *)r;
    } else {
        // all scalar arguments, so just do the operation here
        return PyFloat_FromDouble(f[0] * f[1] + f[2]);
    }
}

PyObject *lerp(PyObject *self, PyObject *args)
{
    int i, l;
    float *arrays[3], f[3];
    int which = 0;
    utility funcs[7] = {
        inLERPvvv,
        inLERPvvs,
        inLERPvsv,
        inLERPvss,
        inLERPsvv,
        inLERPsvs,
        inLERPssv
    };

    for (i = 0; i < 3; i++) {
        PyObject *it = PyTuple_GetItem(args, i);
        if (Vop_Check(it)) {
            l = Vop_truelength(it);
            arrays[i] = ((Vop*)it)->data;
        } else {
            f[i] = PyFloat_AsDouble(it);
            arrays[i] = &f[i];
            which += (4 >> i);
        }
    }
    if (which < 7) {
        Vop *r = (Vop*)Vop_NEW(l);

        odometer += 3 * l;
        (funcs[which])(r->data, arrays, l);
        
        return (PyObject *)r;
    } else {
        double t, a, b;
        if (!PyArg_ParseTuple(args, "fff", &t, &a, &b))
            return 0;
        return PyFloat_FromDouble(a + t * (b - a));
    }
}

PyObject *readodometer(PyObject *self, PyObject *args)
{
    return PyLong_FromLongLong(odometer);
}

static PyMethodDef methods[] = {
  {"array", array, METH_VARARGS},
  {"fromstring", fromstring, METH_VARARGS},
  {"arange", arange, METH_VARARGS},
  {"sqrt", mysqrt, METH_VARARGS},
  {"floor", myfloor, METH_VARARGS},
  {"exp", myexp, METH_VARARGS},
  {"sin", mysin, METH_VARARGS},
  {"cos", mycos, METH_VARARGS},
  {"acos", myacos, METH_VARARGS},
  {"sometrue", sometrue, METH_VARARGS},
  {"alltrue", alltrue, METH_VARARGS},
  {"minimum", minimum, METH_VARARGS},
  {"maximum", maximum, METH_VARARGS},
  {"odometer", readodometer, METH_VARARGS},
  {"compress", compress, METH_VARARGS},
  {"expand", expand, METH_VARARGS},
  {"take", take, METH_VARARGS},
  {"take2", take2, METH_VARARGS},
  {"takeB", takeB, METH_VARARGS},
  {"take2B", take2B, METH_VARARGS},
  {"where", where, METH_VARARGS},
  {"replicate", replicate, METH_VARARGS},
  {"duplicate", duplicate, METH_VARARGS},
  {"lerp2", lerp, METH_VARARGS},
  {"mad", mad, METH_VARARGS},
  {NULL, NULL},
};

/* Module init function */

void initvop()
{
    PyObject *m, *d;

    m = Py_InitModule("vop", methods);
    d = PyModule_GetDict(m);

    /* initialize module variables/constants */

#if PYTHON_API_VERSION >= 1007
    vop_error = PyErr_NewException("vop.error", NULL, NULL);
#else
    vop_error = Py_BuildValue("s", "vop.error");
#endif
    PyDict_SetItemString(d, "error", vop_error);
}
