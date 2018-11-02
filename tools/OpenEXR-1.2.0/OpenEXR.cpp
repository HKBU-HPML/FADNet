#include <Python.h>

#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#endif

#include <ImathBox.h>
#include <ImfArray.h>
#include <ImfAttribute.h>
#include <ImfBoxAttribute.h>
#include <ImfChannelList.h>
#include <ImfStandardAttributes.h>
#include <ImfChannelListAttribute.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfCompressionAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfEnvmapAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfIntAttribute.h>
#include <ImfKeyCodeAttribute.h>
#include <ImfLineOrderAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOutputFile.h>
#include <ImfPreviewImageAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfTileDescriptionAttribute.h>
#include <ImfTiledOutputFile.h>
#include <ImfTimeCodeAttribute.h>
#include <ImfVecAttribute.h>
#include <ImfVersion.h>

#include <iostream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
using namespace Imf;
using namespace Imath;

static PyObject *OpenEXR_error = NULL;
static PyObject *pModuleImath;

static PyObject *PyObject_StealAttrString(PyObject* o, const char *name)
{
    PyObject *r = PyObject_GetAttrString(o, name);
    Py_DECREF(r);
    return r;
}

////////////////////////////////////////////////////////////////////////
//    InputFile
////////////////////////////////////////////////////////////////////////

typedef struct {
    PyObject_HEAD
    InputFile i;
    int is_opened;
} InputFileC;

static PyObject *channel(PyObject *self, PyObject *args, PyObject *kw)
{
    InputFile *file = &((InputFileC *)self)->i;

    Box2i dw = file->header().dataWindow();
    int miny, maxy;

    miny = dw.min.y;
    maxy = dw.max.y;

    char *cname;
    PyObject *pixel_type = NULL;
    char *keywords[] = { (char*)"cname", (char*)"pixel_type", (char*)"scanLine1", (char*)"scanLine2", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|Oii", keywords, &cname, &pixel_type, &miny, &maxy))
        return NULL;

    if (maxy < miny) {
        PyErr_SetString(PyExc_TypeError, "scanLine1 must be <= scanLine2");
        return NULL;
    }
    if (miny < dw.min.y) {
        PyErr_SetString(PyExc_TypeError, "scanLine1 cannot be outside dataWindow");
        return NULL;
    }
    if (maxy > dw.max.y) {
        PyErr_SetString(PyExc_TypeError, "scanLine2 cannot be outside dataWindow");
        return NULL;
    }

    ChannelList channels = file->header().channels();
    Channel *channelPtr = channels.findChannel(cname);
    if (channelPtr == NULL) {
        return PyErr_Format(PyExc_TypeError, "There is no channel '%s' in the image", cname);
    }

    Imf::PixelType pt;
    if (pixel_type != NULL) {
        pt = PixelType(PyLong_AsLong(PyObject_StealAttrString(pixel_type, "v")));
    } else {
        pt = channelPtr->type;
    }

    int width  = dw.max.x - dw.min.x + 1;
    int height = maxy - miny + 1;

    size_t typeSize;
    switch (pt) {
    case HALF:
        typeSize = 2;
        break;

    case FLOAT:
    case UINT:
        typeSize = 4;
        break;

    default:
        PyErr_SetString(PyExc_TypeError, "Unknown type");
        return NULL;
    }
    PyObject *r = PyString_FromStringAndSize(NULL, typeSize * width * height);

    char *pixels = PyString_AsString(r);

    try
    {
        FrameBuffer frameBuffer;
        size_t xstride = typeSize;
        size_t ystride = typeSize * width;
        frameBuffer.insert(cname,
                           Slice(pt,
                                 pixels - dw.min.x * xstride - miny * ystride,
                                 xstride,
                                 ystride,
                                 1,1,
                                 0.0));
        file->setFrameBuffer(frameBuffer);
        file->readPixels(miny, maxy);
    }
    catch (const std::exception &e)
    {
       PyErr_SetString(PyExc_IOError, e.what());
       return NULL;
    }

    return r;
}

static PyObject *channels(PyObject *self, PyObject *args, PyObject *kw)
{
    InputFile *file = &((InputFileC *)self)->i;

    Box2i dw = file->header().dataWindow();
    int miny, maxy;

    miny = dw.min.y;
    maxy = dw.max.y;

    PyObject *clist;
    PyObject *pixel_type = NULL;
    char *keywords[] = { (char*)"cnames", (char*)"pixel_type", (char*)"scanLine1", (char*)"scanLine2", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|Oii", keywords, &clist, &pixel_type, &miny, &maxy))
        return NULL;

    if (maxy < miny) {
        PyErr_SetString(PyExc_TypeError, "scanLine1 must be <= scanLine2");
        return NULL;
    }
    if (miny < dw.min.y) {
        PyErr_SetString(PyExc_TypeError, "scanLine1 cannot be outside dataWindow");
        return NULL;
    }
    if (maxy > dw.max.y) {
        PyErr_SetString(PyExc_TypeError, "scanLine2 cannot be outside dataWindow");
        return NULL;
    }

    ChannelList channels = file->header().channels();
    FrameBuffer frameBuffer;

    int width  = dw.max.x - dw.min.x + 1;
    int height = maxy - miny + 1;

    PyObject *retval = PyList_New(0);
    PyObject *iterator = PyObject_GetIter(clist);
    if (iterator == NULL) {
      PyErr_SetString(PyExc_TypeError, "Channel list must be iterable");
      return NULL;
    }
    PyObject *item;

    while ((item = PyIter_Next(iterator)) != NULL) {
      char *cname = PyString_AsString(item);
      Channel *channelPtr = channels.findChannel(cname);
      if (channelPtr == NULL) {
          return PyErr_Format(PyExc_TypeError, "There is no channel '%s' in the image", cname);
      }

      Imf::PixelType pt;
      if (pixel_type != NULL) {
          pt = PixelType(PyLong_AsLong(PyObject_StealAttrString(pixel_type, "v")));
      } else {
          pt = channelPtr->type;
      }

      // Use pt to compute typeSize
      size_t typeSize;
      switch (pt) {
      case HALF:
          typeSize = 2;
          break;

      case FLOAT:
      case UINT:
          typeSize = 4;
          break;

      default:
          PyErr_SetString(PyExc_TypeError, "Unknown type");
          return NULL;
      }

      size_t xstride = typeSize;
      size_t ystride = typeSize * width;

      PyObject *r = PyString_FromStringAndSize(NULL, typeSize * width * height);
      PyList_Append(retval, r);
      Py_DECREF(r);

      char *pixels = PyString_AsString(r);

      try
      {
          frameBuffer.insert(cname,
                             Slice(pt,
                                   pixels - dw.min.x * xstride - miny * ystride,
                                   xstride,
                                   ystride,
                                   1,1,
                                   0.0));
      }
      catch (const std::exception &e)
      {
         PyErr_SetString(PyExc_IOError, e.what());
         return NULL;
      }
      Py_DECREF(item);
    }
    Py_DECREF(iterator);
    file->setFrameBuffer(frameBuffer);
    file->readPixels(miny, maxy);

    return retval;
}
static PyObject *inclose(PyObject *self, PyObject *args)
{
  InputFileC *pc = ((InputFileC *)self);
  if (pc->is_opened) {
    pc->is_opened = 0;
    InputFile *file = &((InputFileC *)self)->i;
    file->~InputFile();
  }
  Py_RETURN_NONE;
}

static PyObject *dict_from_header(Header h)
{
    PyObject *object;

    object = PyDict_New();

    PyObject *pV2FFunc = PyObject_GetAttrString(pModuleImath, "V2f");
    PyObject *pChanFunc = PyObject_GetAttrString(pModuleImath, "Channel");
    PyObject *pPTFunc = PyObject_GetAttrString(pModuleImath, "PixelType");
    PyObject *pBoxFunc = PyObject_GetAttrString(pModuleImath, "Box2i");
    PyObject *pPointFunc = PyObject_GetAttrString(pModuleImath, "point");
    PyObject *pPIFunc = PyObject_GetAttrString(pModuleImath, "PreviewImage");
    PyObject *pLOFunc = PyObject_GetAttrString(pModuleImath, "LineOrder");
    PyObject *pCFunc = PyObject_GetAttrString(pModuleImath, "Compression");
    PyObject *pCHFunc = PyObject_GetAttrString(pModuleImath, "chromaticity");
    PyObject *pCHSFunc = PyObject_GetAttrString(pModuleImath, "Chromaticities");

    for (Header::ConstIterator i = h.begin(); i != h.end(); ++i) {
        const Attribute *a = &i.attribute();
        PyObject *ob = NULL;

        // cout << i.name() << " (type " << a->typeName() << ")\n";
        if (const Box2iAttribute *ta = dynamic_cast <const Box2iAttribute *> (a)) {

            PyObject *ptargs[2];
            ptargs[0] = Py_BuildValue("ii", ta->value().min.x, ta->value().min.y);
            ptargs[1] = Py_BuildValue("ii", ta->value().max.x, ta->value().max.y);
            PyObject *pt[2];
            pt[0] = PyObject_CallObject(pPointFunc, ptargs[0]);
            pt[1] = PyObject_CallObject(pPointFunc, ptargs[1]);
            PyObject *boxArgs = Py_BuildValue("NN", pt[0], pt[1]);

            ob = PyObject_CallObject(pBoxFunc, boxArgs);
            Py_DECREF(boxArgs);
            Py_DECREF(ptargs[0]);
            Py_DECREF(ptargs[1]);
        } else if (const PreviewImageAttribute *pia = dynamic_cast <const PreviewImageAttribute *> (a)) {
            int size = pia->value().width() * pia->value().height() * 4;
            PyObject *args = Py_BuildValue("iis#", pia->value().width(), pia->value().height(), (char*)pia->value().pixels(), size);
            ob = PyObject_CallObject(pPIFunc, args);
            Py_DECREF(args);
        } else if (const LineOrderAttribute *ta = dynamic_cast <const LineOrderAttribute *> (a)) {
            PyObject *args = PyTuple_Pack(1, PyInt_FromLong(ta->value()));
            ob = PyObject_CallObject(pLOFunc, args);
            Py_DECREF(args);
        } else if (const CompressionAttribute *ta = dynamic_cast <const CompressionAttribute *> (a)) {
            PyObject *args = PyTuple_Pack(1, PyInt_FromLong(ta->value()));
            ob = PyObject_CallObject(pCFunc, args);
            Py_DECREF(args);
        } else
        if (const ChannelListAttribute *ta = dynamic_cast <const ChannelListAttribute *> (a)) {
            const ChannelList cl = ta->value();
            PyObject *CS = PyDict_New();
            for (ChannelList::ConstIterator j = cl.begin(); j != cl.end(); ++j) {
                PyObject *ptarg = Py_BuildValue("(i)", j.channel().type);
                PyObject *pt = PyObject_CallObject(pPTFunc, ptarg);
                PyObject *chanarg = Py_BuildValue("Nii",
                    pt,
                    j.channel().xSampling,
                    j.channel().ySampling);
                PyObject *C = PyObject_CallObject(pChanFunc, chanarg);
                PyDict_SetItemString(CS, j.name(), C);
                Py_DECREF(C);
                Py_DECREF(ptarg);
                Py_DECREF(chanarg);
            }
            ob = CS;
        } else
        if (const FloatAttribute *ta = dynamic_cast <const FloatAttribute *> (a)) {
            ob = PyFloat_FromDouble(ta->value());
        } else if (const IntAttribute *ta = dynamic_cast <const IntAttribute *> (a)) {
            ob = PyInt_FromLong(ta->value());
        } else if (const V2fAttribute *ta = dynamic_cast <const V2fAttribute *> (a)) {
            PyObject *args = Py_BuildValue("ff", ta->value().x, ta->value().y);
            ob = PyObject_CallObject(pV2FFunc, args);
            Py_DECREF(args);
        } else
        if (const StringAttribute *ta = dynamic_cast <const StringAttribute *> (a)) {
            ob = PyString_FromString(ta->value().c_str());
        } else
        if (const ChromaticitiesAttribute *ta = dynamic_cast<const ChromaticitiesAttribute *>(a)) {
            const Chromaticities &ch(ta->value());
            PyObject *rgbwargs[4];
            rgbwargs[0] = Py_BuildValue("ff", ch.red[0], ch.red[1]);
            rgbwargs[1] = Py_BuildValue("ff", ch.green[0], ch.green[1]);
            rgbwargs[2] = Py_BuildValue("ff", ch.blue[0], ch.blue[1]);
            rgbwargs[3] = Py_BuildValue("ff", ch.white[0], ch.white[1]);
            PyObject *chromas[4];
            chromas[0] = PyObject_CallObject(pCHFunc, rgbwargs[0]);
            chromas[1] = PyObject_CallObject(pCHFunc, rgbwargs[1]);
            chromas[2] = PyObject_CallObject(pCHFunc, rgbwargs[2]);
            chromas[3] = PyObject_CallObject(pCHFunc, rgbwargs[3]);
            PyObject *cargs = Py_BuildValue("NNNN", chromas[0], chromas[1], chromas[2], chromas[3]);
            ob = PyObject_CallObject(pCHSFunc, cargs);
            Py_DECREF(cargs);
            Py_DECREF(rgbwargs[0]);
            Py_DECREF(rgbwargs[1]);
            Py_DECREF(rgbwargs[2]);
            Py_DECREF(rgbwargs[3]);
#ifdef INCLUDED_IMF_STRINGVECTOR_ATTRIBUTE_H
        } else if (const StringVectorAttribute *ta = dynamic_cast<const StringVectorAttribute *>(a)) {
            StringVector sv = ta->value();
            ob = PyList_New(sv.size());
            for (size_t i = 0; i < sv.size(); i++)
                PyList_SetItem(ob, i, PyString_FromString(sv[i].c_str()));
#endif
        } else {
            // Unknown type for this object, so set its value to None.
            // printf("Baffled by type %s\n", a->typeName());
            ob = Py_None;
            Py_INCREF(ob);
        }
        PyDict_SetItemString(object, i.name(), ob);
        Py_DECREF(ob);
    }

    Py_DECREF(pV2FFunc);
    Py_DECREF(pChanFunc);
    Py_DECREF(pPTFunc);
    Py_DECREF(pBoxFunc);
    Py_DECREF(pPointFunc);
    Py_DECREF(pPIFunc);
    Py_DECREF(pLOFunc);
    Py_DECREF(pCFunc);

    return object;
}

static PyObject *inheader(PyObject *self, PyObject *args)
{
    InputFile *file = &((InputFileC *)self)->i;
    return dict_from_header(file->header());
}

static PyObject *isComplete(PyObject *self, PyObject *args)
{
    InputFile *file = &((InputFileC *)self)->i;
    return PyBool_FromLong(file->isComplete());
}

/* Method table */
static PyMethodDef InputFile_methods[] = {
  {"header", inheader, METH_VARARGS},
  {"channel", (PyCFunction)channel, METH_KEYWORDS},
  {"channels", (PyCFunction)channels, METH_KEYWORDS},
  {"close", inclose, METH_VARARGS},
  {"isComplete", isComplete, METH_VARARGS},
  {NULL, NULL},
};

static void
InputFile_dealloc(PyObject *self)
{
    Py_DECREF(inclose(self, NULL));
    PyObject_Del(self);
}

static PyObject *
InputFile_GetAttr(PyObject *self, char *attrname)
{
    return Py_FindMethod(InputFile_methods, self, attrname);
}

static PyObject *
InputFile_Repr(PyObject *self)
{
    //PyObject *result = NULL;
    char buf[50];


    sprintf(buf, "InputFile represented");
    return PyString_FromString(buf);
}

static PyTypeObject InputFile_Type = {
    PyObject_HEAD_INIT(&PyType_Type)
    0,
    "OpenEXR.InputFile",
    sizeof(InputFileC),
    0,
    (destructor)InputFile_dealloc,
    0,
    (getattrfunc)InputFile_GetAttr,
    0,
    0,
    (reprfunc)InputFile_Repr,
    0, //&InputFile_as_number,
    0, //&InputFile_as_sequence,
    0,

    0,
    0,
    0,
    0,
    0,
    
    0,
    
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    0,
    0,
    0,
    0,

    /* the rest are NULLs */
};

int makeInputFile(PyObject *self, PyObject *args, PyObject *kwds)
{
    InputFileC *object = ((InputFileC *)self);
    char *filename;

    if (!PyArg_ParseTuple(args, "s:InputFile", &filename))
       return -1;

    try
    {
        new(&object->i) InputFile(filename);
    }
    catch (const std::exception &e)
    {
       // Py_DECREF(object);
       PyErr_SetString(PyExc_IOError, e.what());
       return -1;
    }
    object->is_opened = 1;

    return 0;
}


////////////////////////////////////////////////////////////////////////
//    OutputFile
////////////////////////////////////////////////////////////////////////

typedef struct {
    PyObject_HEAD
    OutputFile o;
    int is_opened;
} OutputFileC;

static PyObject *outwrite(PyObject *self, PyObject *args)
{
    OutputFile *file = &((OutputFileC *)self)->o;

    // long height = PyLong_AsLong(PyTuple_GetItem(args, 1));
    Box2i dw = file->header().dataWindow();
    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;
	PyObject *pixeldata;
	
    if (!PyArg_ParseTuple(args, "O!|i:writePixels", &PyDict_Type, &pixeldata, &height))
       return NULL;

    FrameBuffer frameBuffer;

    const ChannelList &channels = file->header().channels();
    for (ChannelList::ConstIterator i = channels.begin();
         i != channels.end();
         ++i) {
        PyObject *channel_spec = PyDict_GetItem(pixeldata, PyString_FromString(i.name()));
        if (channel_spec != NULL) {
            Imf::PixelType pt = i.channel().type;
            int typeSize = 4;
            switch (pt) {
            case HALF:
                typeSize = 2;
                break;

            case FLOAT:
            case UINT:
                typeSize = 4;
                break;

            default:
                break;
            }
            int yStride = typeSize * width;

            if (!PyString_Check(channel_spec)) {
                PyErr_Format(PyExc_TypeError, "Data for channel '%s' must be a string", i.name());
                return NULL;
            }
            if (PyString_Size(channel_spec) != (height * yStride)) {
                PyErr_Format(PyExc_TypeError, "Data for channel '%s' should have size %d but got %zu", i.name(), (height * yStride), PyString_Size(channel_spec));
                return NULL;
            }

            char *srcPixels = PyString_AsString(channel_spec);

            frameBuffer.insert(i.name(),                        // name
                Slice(pt,                                       // type
                      srcPixels - dw.min.x * typeSize - file->currentScanLine() * yStride,                         // base 
                      typeSize,                                 // xStride
                      yStride));                                // yStride
        }
    }

    try
    {
        file->setFrameBuffer(frameBuffer);
        file->writePixels(height);
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(PyExc_IOError, e.what());
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *outcurrentscanline(PyObject *self, PyObject *args)
{
    OutputFile *file = &((OutputFileC *)self)->o;
    return PyLong_FromLong(file->currentScanLine());
}

static PyObject *outclose(PyObject *self, PyObject *args)
{
    OutputFileC *oc = (OutputFileC *)self;
    if (oc->is_opened) {
      oc->is_opened = 0;
      OutputFile *file = &oc->o;
      file->~OutputFile();
    }
    Py_RETURN_NONE;
}

/* Method table */
static PyMethodDef OutputFile_methods[] = {
  {"writePixels", outwrite, METH_VARARGS},
  {"currentScanLine", outcurrentscanline, METH_VARARGS},
  {"close", outclose, METH_VARARGS},
  {NULL, NULL},
};

static void
OutputFile_dealloc(PyObject *self)
{
    Py_DECREF(outclose(self, NULL));
    PyObject_Del(self);
}

static PyObject *
OutputFile_GetAttr(PyObject *self, char *attrname)
{
    return Py_FindMethod(OutputFile_methods, self, attrname);
}

static PyObject *
OutputFile_Repr(PyObject *self)
{
    //PyObject *result = NULL;
    char buf[50];

    sprintf(buf, "OutputFile represented");
    return PyString_FromString(buf);
}

static PyTypeObject OutputFile_Type = {
    PyObject_HEAD_INIT(&PyType_Type)
    0,
    "OpenEXR.OutputFile",
    sizeof(OutputFileC),
    0,
    (destructor)OutputFile_dealloc,
    0,
    (getattrfunc)OutputFile_GetAttr,
    0,
    0,
    (reprfunc)OutputFile_Repr,
    0, //&InputFile_as_number,
    0, //&InputFile_as_sequence,
    0,

    0,
    0,
    0,
    0,
    0,
    
    0,
    
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    0,
    0,
    0,
    0,

    /* the rest are NULLs */
};

int makeOutputFile(PyObject *self, PyObject *args, PyObject *kwds)
{
    char *filename;
    PyObject *header_dict;

    if (!PyArg_ParseTuple(args, "sO!:OutputFile", &filename, &PyDict_Type, &header_dict))
       return -1;

    OutputFileC *object = (OutputFileC *)self;

    Header header(64, 64);

    PyObject *pB2i = PyObject_GetAttrString(pModuleImath, "Box2i");
    PyObject *pB2f = PyObject_GetAttrString(pModuleImath, "Box2f");
    PyObject *pV2f = PyObject_GetAttrString(pModuleImath, "V2f");
    PyObject *pLO = PyObject_GetAttrString(pModuleImath, "LineOrder");
    PyObject *pCOMP = PyObject_GetAttrString(pModuleImath, "Compression");
    PyObject *pPI = PyObject_GetAttrString(pModuleImath, "PreviewImage");
    PyObject *pCH = PyObject_GetAttrString(pModuleImath, "Chromaticities");

    Py_ssize_t pos = 0;
    PyObject *key, *value;

    while (PyDict_Next(header_dict, &pos, &key, &value)) {
        if (PyFloat_Check(value)) {
            header.insert(PyString_AsString(key), FloatAttribute(PyFloat_AsDouble(value)));
        }
        else if (PyInt_Check(value)) {
            header.insert(PyString_AsString(key), IntAttribute(PyInt_AsLong(value)));
        } else if (PyString_Check(value)) {
            header.insert(PyString_AsString(key), StringAttribute(PyString_AsString(value)));
        } else if (PyObject_IsInstance(value, pB2i)) {
            Box2i box(V2i(PyLong_AsLong(PyObject_StealAttrString(PyObject_StealAttrString(value, "min"), "x")),
                          PyLong_AsLong(PyObject_StealAttrString(PyObject_StealAttrString(value, "min"), "y"))),
                      V2i(PyLong_AsLong(PyObject_StealAttrString(PyObject_StealAttrString(value, "max"), "x")),
                          PyLong_AsLong(PyObject_StealAttrString(PyObject_StealAttrString(value, "max"), "y"))));
            header.insert(PyString_AsString(key), Box2iAttribute(box));
        } else if (PyObject_IsInstance(value, pB2f)) {
            Box2f box(V2f(PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "min"), "x")),
                          PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "min"), "y"))),
                      V2f(PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "max"), "x")),
                          PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "max"), "y"))));
            header.insert(PyString_AsString(key), Box2fAttribute(box));
        } else if (PyObject_IsInstance(value, pPI)) {
            PreviewImage pi(PyLong_AsLong(PyObject_StealAttrString(value, "width")),
                            PyLong_AsLong(PyObject_StealAttrString(value, "height")),
                            (Imf::PreviewRgba *)PyString_AsString(PyObject_StealAttrString(value, "pixels")));
            header.insert(PyString_AsString(key), PreviewImageAttribute(pi));
        } else if (PyObject_IsInstance(value, pV2f)) {
            V2f v(PyFloat_AsDouble(PyObject_StealAttrString(value, "x")), PyFloat_AsDouble(PyObject_StealAttrString(value, "y")));

            header.insert(PyString_AsString(key), V2fAttribute(v));
        } else if (PyObject_IsInstance(value, pLO)) {
            LineOrder i = (LineOrder)PyInt_AsLong(PyObject_StealAttrString(value, "v"));

            header.insert(PyString_AsString(key), LineOrderAttribute(i));
        } else if (PyObject_IsInstance(value, pCOMP)) {
            Compression i = (Compression)PyInt_AsLong(PyObject_StealAttrString(value, "v"));

            header.insert(PyString_AsString(key), CompressionAttribute(i));
        } else if (PyObject_IsInstance(value, pCH)) {
            V2f red(PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "red"), "x")),
                    PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "red"), "y")));
            V2f green(PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "green"), "x")),
                      PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "green"), "y")));
            V2f blue(PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "blue"), "x")),
                     PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "blue"), "y")));
            V2f white(PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "white"), "x")),
                      PyFloat_AsDouble(PyObject_StealAttrString(PyObject_StealAttrString(value, "white"), "y")));
            Chromaticities c(red, green, blue, white);
            header.insert(PyString_AsString(key), ChromaticitiesAttribute(c));
        } else if (PyDict_Check(value)) {
            PyObject *key2, *value2;
            Py_ssize_t pos2 = 0;

            while (PyDict_Next(value, &pos2, &key2, &value2)) {
                if (0)
                    printf("%s -> %s\n",
                        PyString_AsString(key2),
                        PyString_AsString(PyObject_Str(PyObject_Type(value2))));
                header.channels().insert(PyString_AsString(key2),
                                         Channel(PixelType(PyLong_AsLong(PyObject_StealAttrString(PyObject_StealAttrString(value2, "type"), "v"))),
                                                 PyLong_AsLong(PyObject_StealAttrString(value2, "xSampling")),
                                                 PyLong_AsLong(PyObject_StealAttrString(value2, "ySampling"))));
            }
#ifdef INCLUDED_IMF_STRINGVECTOR_ATTRIBUTE_H
        } else if (PyList_Check(value)) {
            StringVector sv(PyList_Size(value));
            for (size_t i = 0; i < sv.size(); i++)
                sv[i] = PyString_AsString(PyList_GetItem(value, i));
            header.insert(PyString_AsString(key), StringVectorAttribute(sv));
#endif
        } else {
            printf("XXX - unknown attribute: %s\n", PyString_AsString(PyObject_Str(key)));
        }
    }

    Py_DECREF(pB2i);
    Py_DECREF(pB2f);
    Py_DECREF(pV2f);
    Py_DECREF(pLO);
    Py_DECREF(pCOMP);
    Py_DECREF(pPI);

    try
    {
        new(&object->o) OutputFile(filename, header);
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(PyExc_IOError, e.what());
        return -1;
    }
    object->is_opened = 1;
    return 0;
}

////////////////////////////////////////////////////////////////////////

PyObject *makeHeader(PyObject *self, PyObject *args)
{
    int w, h;
    if (!PyArg_ParseTuple(args, "ii:Header", &w, &h))
      return NULL;
    Header header(w, h);
    header.channels().insert("R", Channel(FLOAT));
    header.channels().insert("G", Channel(FLOAT));
    header.channels().insert("B", Channel(FLOAT));
    return dict_from_header(header);
}

////////////////////////////////////////////////////////////////////////

static bool 
isOpenExrFile (const char fileName[]) 
{ 
    std::ifstream f (fileName, std::ios_base::binary); 
    char bytes[4]; 
    f.read (bytes, sizeof (bytes)); 
    return !!f && Imf::isImfMagic (bytes); 
} 


PyObject *_isOpenExrFile(PyObject *self, PyObject *args)
{
    char *filename;
    if (!PyArg_ParseTuple(args, "s:isOpenExrFile", &filename))
        return NULL;
    return PyBool_FromLong(isOpenExrFile(filename));
}

////////////////////////////////////////////////////////////////////////

static PyMethodDef methods[] = {
    {"Header", makeHeader, METH_VARARGS},
    {"isOpenExrFile", _isOpenExrFile, METH_VARARGS},
    {NULL, NULL},
};

extern "C" void initOpenEXR()
{
    PyObject *m, *d, *item;

    Imf::staticInitialize();

    m = Py_InitModule("OpenEXR", methods);
    d = PyModule_GetDict(m);

    pModuleImath = PyImport_Import(item= PyString_FromString("Imath")); Py_DECREF(item);

    /* initialize module variables/constants */
    InputFile_Type.tp_new = PyType_GenericNew;
    InputFile_Type.tp_init = makeInputFile;
    OutputFile_Type.tp_new = PyType_GenericNew;
    OutputFile_Type.tp_init = makeOutputFile;
    if (PyType_Ready(&InputFile_Type) != 0)
        return;
    if (PyType_Ready(&OutputFile_Type) != 0)
        return;
    PyModule_AddObject(m, "InputFile", (PyObject *)&InputFile_Type);
    PyModule_AddObject(m, "OutputFile", (PyObject *)&OutputFile_Type);

#if PYTHON_API_VERSION >= 1007
    OpenEXR_error = PyErr_NewException((char*)"OpenEXR.error", NULL, NULL);
#else
    OpenEXR_error = PyString_FromString("OpenEXR.error");
#endif
    PyDict_SetItemString(d, "error", OpenEXR_error);
    Py_DECREF(OpenEXR_error);

    PyDict_SetItemString(d, "UINT", item= PyLong_FromLong(UINT)); Py_DECREF(item);
    PyDict_SetItemString(d, "HALF", item= PyLong_FromLong(HALF)); Py_DECREF(item);
    PyDict_SetItemString(d, "FLOAT", item= PyLong_FromLong(FLOAT)); Py_DECREF(item);
    PyDict_SetItemString(d, "__version__", item= PyString_FromString(VERSION)); Py_DECREF(item);
}
